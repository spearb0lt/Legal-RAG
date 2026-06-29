Get-WinEvent -LogName Application -MaxEvents 500 -ErrorAction SilentlyContinue |
    Where-Object { $_.ProviderName -eq 'Application Error' -or $_.ProviderName -eq 'Windows Error Reporting' } |
    Where-Object { $_.Message -like '*python*' } |
    Select-Object -First 5 TimeCreated, ProviderName, Id, Message |
    Format-List
