param(
    [ValidateSet('All', 'Precheck', 'Run', 'Inspect')]
    [string]$Phase = 'All'
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$coverageProject = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$coveragePython = 'D:\conda\envs\endtoend-parking\python.exe'
$coveragePrepared = 'runs/prepared/wheeled-coverage-20261104-v1/config.json'
$coverageOutput = 'runs/verification/wheeled-coverage-formal-20261104-v1'
$coverageTask = 'LT-20261104-46'

Push-Location -LiteralPath $coverageProject
try {
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    & $coveragePython -X utf8 -c "import platform,numpy; assert platform.python_version()=='3.12.13' and numpy.__version__=='2.5.2', '环境版本与冻结计划不一致'"
    if ($LASTEXITCODE -ne 0) { throw '环境检查失败' }
    if ((Get-FileHash -LiteralPath $coveragePrepared -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'e319cf5bdfc830227b11ef5791f5f0024da4447b18536ef416ee43f76ac741b9') {
        throw 'prepared身份漂移；停止，不修改SHA绕过'
    }
    & $coveragePython -X utf8 scripts/calibrate_wheeled_tasks.py --phase precheck --prepared $coveragePrepared
    if ($LASTEXITCODE -ne 0) { throw '冻结源/代码/分母前检失败' }
    if ($Phase -eq 'Precheck') { return }

    if ($Phase -in @('All', 'Run')) {
        foreach ($coverageExisting in @($coverageOutput, ($coverageOutput + '-run.log'), ($coverageOutput + '-run-exit.json'), ($coverageOutput + '-inspect.log'), ($coverageOutput + '-inspect-exit.json'))) {
            if (Test-Path -LiteralPath $coverageExisting) { throw "已有现场，禁止覆盖或自动续跑：$coverageExisting" }
        }
        & $coveragePython -X utf8 scripts/calibrate_wheeled_tasks.py --phase run --prepared $coveragePrepared --output $coverageOutput 2>&1 | Tee-Object -FilePath ($coverageOutput + '-run.log')
        $coverageRunExit = $LASTEXITCODE
        [ordered]@{ task=$coverageTask; phase='run'; exit_code=$coverageRunExit } | ConvertTo-Json | Set-Content -LiteralPath ($coverageOutput + '-run-exit.json') -Encoding utf8
        if ($coverageRunExit -ne 0) { throw '全源运行异常；保留现场，不继续inspect或重跑' }
        if ($Phase -eq 'Run') { return }
    }

    if ($Phase -in @('All', 'Inspect')) {
        $coverageRunRecord = Get-Content -LiteralPath ($coverageOutput + '-run-exit.json') -Raw | ConvertFrom-Json
        if ($coverageRunRecord.task -ne $coverageTask -or $coverageRunRecord.phase -ne 'run' -or $coverageRunRecord.exit_code -ne 0) {
            throw '缺少本任务成功完成的run退出记录'
        }
        foreach ($coverageExisting in @(($coverageOutput + '-inspect.log'), ($coverageOutput + '-inspect-exit.json'))) {
            if (Test-Path -LiteralPath $coverageExisting) { throw "已有审核现场，禁止覆盖：$coverageExisting" }
        }
        & $coveragePython -X utf8 scripts/calibrate_wheeled_tasks.py --phase inspect --prepared $coveragePrepared --output $coverageOutput 2>&1 | Tee-Object -FilePath ($coverageOutput + '-inspect.log')
        $coverageInspectExit = $LASTEXITCODE
        [ordered]@{ task=$coverageTask; phase='inspect'; exit_code=$coverageInspectExit } | ConvertTo-Json | Set-Content -LiteralPath ($coverageOutput + '-inspect-exit.json') -Encoding utf8
        if ($coverageInspectExit -ne 0) { throw '全源独立审核失败；保留现场并反馈' }
        $coverageSummary = Get-Content -LiteralPath ($coverageOutput + '-inspect.log') -Raw | ConvertFrom-Json
        if ($coverageSummary.verified -ne $true -or $coverageSummary.cases -ne 2700) { throw '审核未覆盖完整2700分母' }
        Write-Host "LT46已完成：通过 $($coverageSummary.passed)/2700；gate_passed=$($coverageSummary.gate_passed)。保留全部产物，交Agent分析；本脚本不训练。"
    }
}
finally {
    Pop-Location
}
