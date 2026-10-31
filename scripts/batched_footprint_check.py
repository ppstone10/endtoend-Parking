"""RUN-REFINE-001: vectorize the existing sampled footprint oracle.

This is a private experiment kernel, not a different collision model. Geometry
near a floating-point decision boundary, custom checkers and custom dynamics
delegate to the original oracle. The controller also validates its final output
through the scalar full-horizon checker.
"""
from __future__ import annotations

import numpy as np

from planner import RectangleFootprintCollisionChecker
from sim import CircleObstacle, DifferentialDriveModel, PolygonObstacle, RectangleObstacle


class BatchedSafety:
    def __init__(self, checker, vehicle_model, dt, max_v, max_omega):
        self.checker = checker
        self.model = vehicle_model
        self.dt, self.max_v, self.max_omega = dt, max_v, max_omega

    def poses_free(self, poses):
        poses = np.asarray(poses, dtype=np.float64)
        if poses.ndim != 2 or poses.shape[1] != 3:
            raise ValueError('poses must have shape (N, 3)')
        if type(self.checker) is not RectangleFootprintCollisionChecker:
            return np.array([np.isfinite(p).all() and self.checker.pose_free(*p) for p in poses])
        # Read current map geometry on every call: no stale map/obstacle cache.
        checker = self.checker
        x, y, yaw = poses.T
        finite = np.isfinite(poses).all(axis=1)
        safe_yaw = np.where(finite, yaw, 0.)
        c, s = np.cos(safe_yaw), np.sin(safe_yaw)
        hl, hw = checker.half_length, checker.half_width
        ex, ey = abs(c) * hl + abs(s) * hw, abs(s) * hl + abs(c) * hw
        box = np.column_stack((x-ex, x+ex, y-ey, y+ey))
        world = checker.env.world_size / 2.
        free = finite & (box[:, 0] >= -world) & (box[:, 1] <= world) & (box[:, 2] >= -world) & (box[:, 3] <= world)
        # Conservative uncertainty zone only chooses the scalar oracle; it never
        # changes the original 1e-9 contact semantics or the required clearance.
        eps = 1e-8 * max(1., abs(world), hl, hw)
        uncertain = finite & np.any(abs(abs(box) - world) <= eps, axis=1)
        for obstacle in checker.env.obstacles:
            if not obstacle.forbidden:
                continue
            ob = obstacle.bbox
            gaps = np.column_stack((box[:, 1]-ob[0], ob[1]-box[:, 0], box[:, 3]-ob[2], ob[3]-box[:, 2]))
            possible = finite & np.all(gaps >= -eps, axis=1)
            uncertain |= possible & np.any(abs(gaps) <= eps, axis=1)
            idx = np.flatnonzero(free & np.all(gaps >= 0., axis=1))
            if not len(idx):
                continue
            xx, yy, cc, ss = x[idx], y[idx], c[idx], s[idx]
            if type(obstacle) is RectangleObstacle:
                hx, hy = (ob[1]-ob[0])/2., (ob[3]-ob[2])/2.
                dx, dy = (ob[0]+ob[1])/2.-xx, (ob[2]+ob[3])/2.-yy
                margins = np.column_stack((
                    ex[idx]+hx+1e-9-abs(dx), ey[idx]+hy+1e-9-abs(dy),
                    hl+hx*abs(cc)+hy*abs(ss)+1e-9-abs(dx*cc+dy*ss),
                    hw+hx*abs(ss)+hy*abs(cc)+1e-9-abs(-dx*ss+dy*cc)))
                hit = np.all(margins >= 0., axis=1)
                uncertain[idx] |= np.any(abs(margins) <= eps, axis=1)
            elif type(obstacle) is CircleObstacle:
                dx, dy = obstacle.x-xx, obstacle.y-yy
                lx, ly = dx*cc+dy*ss, -dx*ss+dy*cc
                residual = (lx-np.clip(lx, -hl, hl))**2 + (ly-np.clip(ly, -hw, hw))**2-obstacle.radius**2
                hit = residual <= 0.
                uncertain[idx] |= abs(residual) <= eps * max(1., obstacle.radius**2)
            elif type(obstacle) is PolygonObstacle:
                hit, boundary = self._polygon(xx, yy, cc, ss, hl, hw, obstacle.vertices, eps)
                uncertain[idx] |= boundary
            else:
                uncertain[idx] = True
                continue
            free[idx[hit]] = False
        idx = np.flatnonzero(uncertain & finite)
        if len(idx):
            # Initial poses and shared segment endpoints repeat in a population.
            unique, inverse = np.unique(poses[idx], axis=0, return_inverse=True)
            answers = np.array([checker.pose_free(*p) for p in unique], dtype=bool)
            free[idx] = answers[inverse]
        return free

    @staticmethod
    def _polygon(x, y, c, s, hl, hw, vertices, eps):
        vertices = np.asarray(vertices, dtype=np.float64)
        local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
        corners = np.empty((len(x), 4, 2))
        corners[:, :, 0] = x[:, None] + c[:, None]*local[:, 0] - s[:, None]*local[:, 1]
        corners[:, :, 1] = y[:, None] + s[:, None]*local[:, 0] + c[:, None]*local[:, 1]
        dx, dy = vertices[None, :, 0]-x[:, None], vertices[None, :, 1]-y[:, None]
        mx = hl+1e-9-abs(dx*c[:, None]+dy*s[:, None])
        my = hw+1e-9-abs(-dx*s[:, None]+dy*c[:, None])
        hit = np.any((mx >= 0.) & (my >= 0.), axis=1)
        uncertain = np.any(((abs(mx) <= eps) & (my >= -eps)) | ((abs(my) <= eps) & (mx >= -eps)), axis=1)
        px, py = corners[:, :, 0], corners[:, :, 1]
        inside = np.zeros(px.shape, dtype=bool)
        boundary = np.zeros(px.shape, dtype=bool)
        a, ar = corners, np.roll(corners, -1, axis=1)
        def cross(u, v):
            return u[..., 0]*v[..., 1]-u[..., 1]*v[..., 0]
        def on(p, q, r, tolerance):
            return np.all((q >= np.minimum(p, r)-tolerance) & (q <= np.maximum(p, r)+tolerance), axis=-1)
        for j, b in enumerate(vertices):
            br = vertices[(j+1) % len(vertices)]
            edge = br-b
            orient = (px-b[0])*edge[1]-(py-b[1])*edge[0]
            between = (px >= min(b[0], br[0])-1e-9) & (px <= max(b[0], br[0])+1e-9) & (py >= min(b[1], br[1])-1e-9) & (py <= max(b[1], br[1])+1e-9)
            boundary |= (abs(orient) <= 1e-9) & between
            uncertain |= np.any((abs(orient) <= eps) & between, axis=1)
            if edge[1] != 0.:
                crossing = (b[1] > py) != (br[1] > py)
                at_x = b[0]+(py-b[1])*edge[0]/edge[1]
                inside ^= crossing & (at_x > px)
                uncertain |= np.any(crossing & (abs(at_x-px) <= eps), axis=1)
            o1, o2 = cross(ar-a, b-a), cross(ar-a, br-a)
            o3, o4 = cross(edge, a-b), cross(edge, ar-b)
            intersects = ((o1*o2 < 0.) & (o3*o4 < 0.))
            for orientation, p, q, r in ((o1, a, b, ar), (o2, a, br, ar), (o3, b, a, br), (o4, b, ar, br)):
                intersects |= (abs(orientation) <= 1e-9) & on(p, q, r, 1e-9)
                uncertain |= np.any((abs(orientation) <= eps) & on(p, q, r, eps), axis=1)
            hit |= np.any(intersects, axis=1)
        hit |= np.any(inside | boundary, axis=1)
        return hit, uncertain

    def eligible(self, sequences, state, scalar_sequence_safe):
        sequences = np.asarray(sequences, dtype=np.float64)
        if sequences.ndim != 2 or sequences.shape[1] == 0 or sequences.shape[1] % 2:
            raise ValueError('sequences must have shape (N, 2H)')
        if (type(self.model) is not DifferentialDriveModel
                or type(self.checker) is not RectangleFootprintCollisionChecker):
            return np.array([scalar_sequence_safe(s, state) for s in sequences], dtype=bool)
        result = np.zeros(len(sequences), dtype=bool)
        valid = (np.isfinite(sequences).all(axis=1)
                 & np.all(abs(sequences[:, 0::2]) <= self.max_v, axis=1)
                 & np.all(abs(sequences[:, 1::2]) <= self.max_omega, axis=1))
        if not np.isfinite(state.to_array()).all() or not self.checker.pose_free(state.x, state.y, state.yaw):
            return result
        idx = np.flatnonzero(valid)
        if not len(idx):
            return result
        controls = sequences[idx].reshape(len(idx), -1, 2)
        h = controls.shape[1]
        poses = np.empty((len(idx), h+1, 3))
        poses[:, 0] = [state.x, state.y, state.yaw]
        for k in range(h):
            before = poses[:, k]
            v = np.clip(controls[:, k, 0], -self.model.max_v, self.model.max_v)
            w = np.clip(controls[:, k, 1], -self.model.max_omega, self.model.max_omega)
            poses[:, k+1, 0] = before[:, 0]+v*np.cos(before[:, 2])*self.dt
            poses[:, k+1, 1] = before[:, 1]+v*np.sin(before[:, 2])*self.dt
            poses[:, k+1, 2] = before[:, 2]+w*self.dt
        start, end = poses[:, :-1].reshape(-1, 3), poses[:, 1:].reshape(-1, 3)
        delta = end-start
        delta[:, 2] = (delta[:, 2]+np.pi) % (2*np.pi)-np.pi
        ratio = (np.hypot(delta[:, 0], delta[:, 1])+self.checker.corner_radius*abs(delta[:, 2]))/self.checker.resolution
        # Near ceil boundaries, floating-point hypot can change the sample grid.
        ambiguous = abs(ratio-np.rint(ratio)) <= 1e-10
        counts = np.maximum(1, np.ceil(ratio).astype(int))
        if counts.max() > 10000 or not np.isfinite(poses).all():
            return np.array([scalar_sequence_safe(s, state) for s in sequences], dtype=bool)
        j = np.arange(counts.max()+1)
        used = j[None, :] <= counts[:, None]
        t = j[None, :]/counts[:, None]
        sampled = start[:, None, :]+t[:, :, None]*delta[:, None, :]
        sampled[:, :, 2] = (sampled[:, :, 2]+np.pi) % (2*np.pi)-np.pi
        answers = np.ones(used.shape, dtype=bool)
        answers[used] = self.poses_free(sampled[used])
        segment_safe = answers.all(axis=1)
        for i in np.flatnonzero(ambiguous):
            segment_safe[i] = self.checker.swept_segment_free(start[i], end[i])
        result[idx] = segment_safe.reshape(len(idx), h).all(axis=1)
        return result
