"""Inspect GPU availability before starting a local model worker."""
import csv
import subprocess
import time


def gpu_inventory():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"], text=True)
    inventory = {}
    for row in csv.reader(output.splitlines()):
        if len(row) != 3:
            raise ValueError("Unexpected nvidia-smi GPU inventory")
        index, uuid, free = [field.strip() for field in row]
        inventory[index] = {"uuid": uuid, "free_mib": int(free), "pids": []}
    output = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    by_uuid = {item["uuid"]: item for item in inventory.values()}
    for row in csv.reader(output.splitlines()):
        if not row:
            continue
        if len(row) != 2 or row[0].strip() not in by_uuid:
            raise ValueError("Unexpected nvidia-smi process inventory; cannot establish GPU availability")
        by_uuid[row[0].strip()]["pids"].append(int(row[1].strip()))
    return inventory


def wait_for_gpus(group, notify, poll_seconds):
    while True:
        inventory = gpu_inventory()
        if not set(group) <= set(inventory):
            raise ValueError(f"Requested GPU indices are missing: {group}; detected {list(inventory)}")
        selected = {index: inventory[index] for index in group}
        if all(not item["pids"] and item["free_mib"] >= 22 * 1024 for item in selected.values()):
            return selected
        notify("waiting_for_gpus", gpus=selected)
        time.sleep(poll_seconds)
