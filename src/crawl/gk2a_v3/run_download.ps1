# GK-2A NCDC 다운로드 재개 스크립트 (202211~202512, parallel 8)
# 사용: D:\pv_backup\src\crawl\gk2a_v3\run_download.ps1

Set-Location D:\pv_backup

$months = @(
    '202211','202212',
    '202301','202302','202303','202304','202305','202306',
    '202307','202308','202309','202310','202311','202312',
    '202401','202402','202403','202404','202405','202406',
    '202407','202408','202409','202410','202411','202412',
    '202501','202502','202503','202504','202505','202506',
    '202507','202508','202509','202510','202511','202512'
)

Write-Host "=== GK-2A 다운로드 재개 (202211~202512) ===" -ForegroundColor Cyan
Write-Host "  월: $($months.Count)개 (37개월)" -ForegroundColor Gray
Write-Host "  병렬: 8 (높은 속도)" -ForegroundColor Gray
Write-Host ""

& D:\pv_backup\.venv\Scripts\python.exe D:\pv_backup\src\crawl\gk2a_v3\download_nc.py $months --parallel 8
