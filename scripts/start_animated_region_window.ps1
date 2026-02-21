param(
    [string]$OutputPath = "target/perf/wgc-region-workload.json",
    [int]$X = 180,
    [int]$Y = 160,
    [int]$Width = 960,
    [int]$Height = 540,
    [int]$BoxSize = 112,
    [int]$IntervalMs = 8
)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$form = New-Object System.Windows.Forms.Form
$form.StartPosition = [System.Windows.Forms.FormStartPosition]::Manual
$form.Location = New-Object System.Drawing.Point($X, $Y)
$form.ClientSize = New-Object System.Drawing.Size($Width, $Height)
$form.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::None
$form.TopMost = $true
$form.ShowInTaskbar = $false
$form.BackColor = [System.Drawing.Color]::Black

$doubleBufferedProperty = [System.Windows.Forms.Control].GetProperty(
    "DoubleBuffered",
    [System.Reflection.BindingFlags]::Instance -bor [System.Reflection.BindingFlags]::NonPublic
)
if ($null -ne $doubleBufferedProperty) {
    $doubleBufferedProperty.SetValue($form, $true, $null)
}

$state = [PSCustomObject]@{
    Tick = 0
    BoxX = 0
    BoxY = 0
    PrevBoxX = 0
    PrevBoxY = 0
    DirectionX = 1
    DirectionY = 1
}

$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = [Math]::Max(1, $IntervalMs)

$form.Add_Paint({
    param($sender, $eventArgs)

    $graphics = $eventArgs.Graphics
    $clip = $eventArgs.ClipRectangle
    $graphics.FillRectangle([System.Drawing.Brushes]::Black, $clip)

    $r = [int](127 + 127 * [Math]::Sin($state.Tick * 0.090))
    $g = [int](127 + 127 * [Math]::Sin($state.Tick * 0.060 + 2.1))
    $b = [int](127 + 127 * [Math]::Sin($state.Tick * 0.075 + 4.2))
    $color = [System.Drawing.Color]::FromArgb(255, $r, $g, $b)
    $brush = New-Object System.Drawing.SolidBrush($color)
    try {
        $graphics.FillRectangle($brush, $state.BoxX, $state.BoxY, $BoxSize, $BoxSize)
    } finally {
        $brush.Dispose()
    }
})

$timer.Add_Tick({
    $state.PrevBoxX = $state.BoxX
    $state.PrevBoxY = $state.BoxY

    $nextX = $state.BoxX + (6 * $state.DirectionX)
    $nextY = $state.BoxY + (5 * $state.DirectionY)
    $maxX = [Math]::Max(0, $form.ClientSize.Width - $BoxSize)
    $maxY = [Math]::Max(0, $form.ClientSize.Height - $BoxSize)

    if ($nextX -lt 0) {
        $nextX = 0
        $state.DirectionX = 1
    } elseif ($nextX -gt $maxX) {
        $nextX = $maxX
        $state.DirectionX = -1
    }

    if ($nextY -lt 0) {
        $nextY = 0
        $state.DirectionY = 1
    } elseif ($nextY -gt $maxY) {
        $nextY = $maxY
        $state.DirectionY = -1
    }

    $state.BoxX = $nextX
    $state.BoxY = $nextY
    $state.Tick += 1

    $prevRect = New-Object System.Drawing.Rectangle($state.PrevBoxX, $state.PrevBoxY, $BoxSize, $BoxSize)
    $nextRect = New-Object System.Drawing.Rectangle($state.BoxX, $state.BoxY, $BoxSize, $BoxSize)
    $form.Invalidate($prevRect)
    $form.Invalidate($nextRect)
})

$form.Add_Shown({
    $dir = Split-Path -Parent $OutputPath
    if ($dir -and -not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }

    $payload = @{
        x = $form.Left
        y = $form.Top
        width = $form.ClientSize.Width
        height = $form.ClientSize.Height
        hwnd = [int64]$form.Handle
    }

    $payload | ConvertTo-Json -Compress | Set-Content -Path $OutputPath -Encoding ascii
    $timer.Start()
})

$form.Add_FormClosed({
    $timer.Stop()
})

[System.Windows.Forms.Application]::Run($form)
