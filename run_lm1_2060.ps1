# LM 1 (tiny GPT from scratch) on the RTX 2060 only, time-boxed.
#   New run:            .\run_lm1_2060.ps1 -Hours 4
#   Resume after crash: .\run_lm1_2060.ps1 -Resume
#   Continue training:  .\run_lm1_2060.ps1 -ExtendHours 6   (re-warm + new cosine from latest.pt, new seed)
param([double]$Hours = 4, [switch]$Resume, [double]$ExtendHours = 0, [int]$Seed = 1)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path; Set-Location $here
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'      # bus 04 = RTX 2060 = index 0
$env:CUDA_VISIBLE_DEVICES = '0'
Remove-Item train_done.txt -ErrorAction SilentlyContinue
if ($ExtendHours -gt 0) { py -3.12 train.py --resume --extend_hours $ExtendHours --seed $Seed 1>> train.log 2>> train.err }
elseif ($Resume) { py -3.12 train.py --resume --seed $Seed 1>> train.log 2>> train.err }
else { py -3.12 train.py --hours $Hours 1> train.log 2> train.err }
"ALL DONE $(Get-Date)" | Out-File train_done.txt
