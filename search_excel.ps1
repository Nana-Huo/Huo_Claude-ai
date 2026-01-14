$desktopPath = [Environment]::GetFolderPath("Desktop")
$searchPath = Join-Path $desktopPath "zczy工作留痕"
$targetPlate = "宁B76107"

$foundFiles = [System.Collections.ArrayList]@()

Write-Host "Search path: $searchPath" -ForegroundColor Cyan

if (Test-Path $searchPath) {
    $excelFiles = Get-ChildItem -Path $searchPath -Recurse -Include *.xls,*.xlsx -ErrorAction SilentlyContinue
    Write-Host "Searching $($excelFiles.Count) Excel files..." -ForegroundColor Yellow

    foreach ($file in $excelFiles) {
        $excel = $null
        try {
            $excel = New-Object -ComObject Excel.Application
            $excel.Visible = $false
            $excel.DisplayAlerts = $false

            $workbook = $excel.Workbooks.Open($file.FullName)
            $sheet = $workbook.Sheets(1)

            $usedRange = $sheet.UsedRange
            $values = $usedRange.Value2

            if ($values -is [object[,]]) {
                $found = $false
                $rows = $values.GetLength(0)
                $cols = $values.GetLength(1)

                for ($i = 1; $i -le $rows; $i++) {
                    for ($j = 1; $j -le $cols; $j++) {
                        $cellValue = $values[$i, $j]
                        if ($cellValue -ne $null -and $cellValue.ToString() -like "*$targetPlate*") {
                            [void]$foundFiles.Add($file.FullName)
                            Write-Host "Found: $($file.FullName)" -ForegroundColor Green
                            $found = $true
                            break
                        }
                    }
                    if ($found) { break }
                }
            }
            elseif ($values -ne $null -and $values.ToString() -like "*$targetPlate*") {
                [void]$foundFiles.Add($file.FullName)
                Write-Host "Found: $($file.FullName)" -ForegroundColor Green
            }

            $workbook.Close($false)
            $excel.Quit()
        }
        catch {
        }
        finally {
            if ($excel -ne $null) {
                [System.Runtime.Interopservices.Marshal]::ReleaseComObject($excel) | Out-Null
            }
        }
    }
} else {
    Write-Host "Path not found: $searchPath" -ForegroundColor Red
}

Write-Host "`nSearch completed!" -ForegroundColor Cyan
if ($foundFiles.Count -eq 0) {
    Write-Host "No files found containing '$targetPlate'" -ForegroundColor Red
} else {
    Write-Host "Found $($foundFiles.Count) files:" -ForegroundColor Green
    $foundFiles | ForEach-Object { Write-Host $_ }
}