param(
    [string]$OutputFile = "",
    [int]$Width = 1280,
    [int]$Height = 720,
    [int]$SquareSize = 120,
    [int]$TickMs = 16
)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$form = New-Object System.Windows.Forms.Form
$form.Text = "Snow Capture Bench Window"
$form.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen
$form.ClientSize = New-Object System.Drawing.Size($Width, $Height)
$form.BackColor = [System.Drawing.Color]::Black
$form.KeyPreview = $true

$panel = New-Object System.Windows.Forms.Panel
$panel.Dock = [System.Windows.Forms.DockStyle]::Fill
$panel.BackColor = [System.Drawing.Color]::Black
$panelType = $panel.GetType()
$doubleBufferedProp = $panelType.GetProperty("DoubleBuffered", [Reflection.BindingFlags]"NonPublic,Instance")
if ($null -ne $doubleBufferedProp) {
    $doubleBufferedProp.SetValue($panel, $true, $null)
}
$form.Controls.Add($panel)

$script:x = 32
$script:y = 32
$script:vx = 9
$script:vy = 7
$script:hue = 0

$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = [Math]::Max(1, $TickMs)

$panel.Add_Paint({
    param($sender, $e)

    $clip = $e.ClipRectangle
    $e.Graphics.FillRectangle([System.Drawing.Brushes]::Black, $clip)

    $r = [int](128 + 127 * [Math]::Sin($script:hue * 0.05))
    $g = [int](128 + 127 * [Math]::Sin($script:hue * 0.05 + 2.0))
    $b = [int](128 + 127 * [Math]::Sin($script:hue * 0.05 + 4.0))
    $brush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb($r, $g, $b))
    $e.Graphics.FillRectangle($brush, $script:x, $script:y, $SquareSize, $SquareSize)
    $brush.Dispose()
})

$timer.Add_Tick({
    $oldX = $script:x
    $oldY = $script:y

    $script:x += $script:vx
    $script:y += $script:vy

    $maxX = [Math]::Max(0, $panel.ClientSize.Width - $SquareSize)
    $maxY = [Math]::Max(0, $panel.ClientSize.Height - $SquareSize)

    if ($script:x -lt 0) {
        $script:x = 0
        $script:vx = -$script:vx
    } elseif ($script:x -gt $maxX) {
        $script:x = $maxX
        $script:vx = -$script:vx
    }

    if ($script:y -lt 0) {
        $script:y = 0
        $script:vy = -$script:vy
    } elseif ($script:y -gt $maxY) {
        $script:y = $maxY
        $script:vy = -$script:vy
    }

    $script:hue = ($script:hue + 3) % 360

    $oldRect = New-Object System.Drawing.Rectangle($oldX, $oldY, $SquareSize, $SquareSize)
    $newRect = New-Object System.Drawing.Rectangle($script:x, $script:y, $SquareSize, $SquareSize)

    $panel.Invalidate($oldRect)
    $panel.Invalidate($newRect)
})

$form.Add_Shown({
    $timer.Start()
    if (-not [string]::IsNullOrWhiteSpace($OutputFile)) {
        $hwnd = $form.Handle.ToInt64()
        try {
            $directory = [System.IO.Path]::GetDirectoryName($OutputFile)
            if (-not [string]::IsNullOrWhiteSpace($directory)) {
                [System.IO.Directory]::CreateDirectory($directory) | Out-Null
            }
            [System.IO.File]::WriteAllText($OutputFile, ("0x{0:X}" -f $hwnd))
        } catch {
            Write-Warning ("Failed to write HWND output file '{0}': {1}" -f $OutputFile, $_.Exception.Message)
        }
    }
})

$form.Add_FormClosing({
    $timer.Stop()
    $timer.Dispose()
})

$form.Add_KeyDown({
    param($sender, $eventArgs)
    if ($eventArgs.KeyCode -eq [System.Windows.Forms.Keys]::Escape) {
        $form.Close()
    }
})

[System.Windows.Forms.Application]::Run($form)
