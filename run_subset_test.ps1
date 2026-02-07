$ContainerName = "subset-test-cont"
$ImageName = "riscv-opt-env"
$HostDir = $PSScriptRoot
$WorkDir = "/work"

Write-Host "Checking Docker container: $ContainerName..."

# Check if container exists
$ContainerStatus = docker ps -a --filter "name=^/${ContainerName}$" --format "{{.Status}}"

if (-not $ContainerStatus) {
    Write-Host "Container does not exist. Creating and starting..."
    docker run -itd --name $ContainerName -v "${HostDir}:${WorkDir}" $ImageName
} elseif ($ContainerStatus -like "Exited*") {
    Write-Host "Container is stopped. Starting..."
    docker start $ContainerName
} else {
    Write-Host "Container is already running."
}

Write-Host "Executing subset test inside container..."
docker exec -w /work/sbllm_repo $ContainerName ./run_subset_test.sh
