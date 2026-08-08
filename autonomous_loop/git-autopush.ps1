# Git Auto-Push Script for Engine_1
# This script polls for changes every 5 seconds and automatically pushes to GitHub.

$debounceSeconds = 5
$ignoredPatterns = @("Seeding", "catboost_info", "__pycache__", "\.git", "\.ai", "\.trash", "\.xlsx$", "\.zip$", "\.png$", "\.pdf$", "\.tmp$")

Write-Host "Initializing Git Auto-Push daemon inside: $PWD"
Write-Host "Make sure you have configured a remote repository first: git remote add origin <your-repo-url>"

while ($true) {
    # Check if there are changes inside the current directory
    $changes = git status --porcelain .
    
    if ($changes) {
        # Filter out ignored patterns
        $filteredChanges = @()
        foreach ($line in $changes) {
            $isIgnored = $false
            foreach ($pattern in $ignoredPatterns) {
                if ($line -match $pattern) {
                    $isIgnored = $true
                    break
                }
            }
            if (-not $isIgnored) {
                $filteredChanges += $line
            }
        }

        if ($filteredChanges.Count -gt 0) {
            Write-Host "Change detected! Waiting $debounceSeconds seconds to debounce..."
            Start-Sleep -Seconds $debounceSeconds
            
            Write-Host "Staging changes..."
            git add .
            
            Write-Host "Committing changes..."
            git commit -m "Auto-update code files [$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')]"
            
            Write-Host "Pushing to GitHub..."
            git push origin master
            
            Write-Host "Sync complete. Listening for new changes..."
        }
    }
    
    Start-Sleep -Seconds 5
}
