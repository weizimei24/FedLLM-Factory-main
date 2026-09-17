<#!
Run the minimal MMLU knowledge-transfer comparison for a single domain.

Default: mathematics, evaluated for base, the corresponding local adapter,
and centralized.  The Python evaluator uses candidate log-likelihood rather
than free-form generation, and writes all outputs under results/stage1.
#>
param(
    [ValidateSet("mathematics", "physics", "computer_science", "statistics", "economics", "biology")]
    [string]$Domain = "mathematics",
    [int]$BatchSize = 8,
    [switch]$Offline
)

$ErrorActionPreference = "Stop"
$arguments = @("-m", "stage1.eval_mmlu", "--domain", $Domain, "--methods", "base,local,centralized", "--batch-size", $BatchSize)
if ($Offline) {
    $arguments += "--offline"
}
& python @arguments
