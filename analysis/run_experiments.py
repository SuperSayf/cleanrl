import csv
import time
import paramiko
import re
import json
import socket
import datetime
import logging
from colorama import init, Fore, Style

# Configure logging
logging.basicConfig(filename='experiment_manager.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

# Initialize colorama
init(autoreset=True)

ssh_details = {
    "hostname": "",
    "username": "sjumoorty2",
    "password": "",
}

# Read experiments from the file
def load_experiments(filename):
    experiments = []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            experiments.append(
                {
                    "grayscale": row["grayscale"] == "True",
                    "use_compression": row["use_compression"] == "True",
                    "jpeg_quality": int(row["jpeg_quality"]),
                    "resolution_width": int(row["resolution_width"]),
                    "resolution_height": int(row["resolution_height"]),
                    "frame_skip": int(row["frame_skip"]),
                }
            )
    return experiments

def print_colored(message, color=Fore.WHITE, style=Style.NORMAL):
    print(f"{style}{color}{message}{Style.RESET_ALL}")
    logging.info(message)

def ssh_connect_with_retry(ssh_details, max_retries=5, delay=60):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    for attempt in range(max_retries):
        try:
            print_colored(f"Attempting SSH connection (attempt {attempt + 1}/{max_retries})...", Fore.YELLOW)
            ssh.connect(**ssh_details)
            print_colored("SSH connection established successfully.", Fore.GREEN)
            return ssh
        except Exception as e:
            print_colored(f"SSH connection failed: {str(e)}", Fore.RED)
            logging.error(f"SSH connection failed: {str(e)}")
            if attempt < max_retries - 1:
                print_colored(f"Retrying in {delay} seconds...", Fore.YELLOW)
                time.sleep(delay)
            else:
                print_colored("Max retries reached. Unable to establish SSH connection.", Fore.RED)
                raise
    return None

def should_avoid_node(node_name, nodes_to_avoid):
    match = re.match(r"mscluster(\d+)", node_name)
    if match:
        node_number = int(match.group(1))
        return node_number in nodes_to_avoid
    return False

def get_available_nodes(ssh, nodes_to_avoid):
    partitions = ["bigbatch", "stampede"]
    available_nodes = {}

    for partition in partitions:
        try:
            stdin, stdout, stderr = ssh.exec_command(f"sinfo -p {partition} -h -o '%n %t'")
            output = stdout.read().decode().strip()
            idle_nodes = 0
            for line in output.split("\n"):
                if line.strip():  # Check if line is not empty
                    node_name, state = line.split()
                    if state == "idle" and not should_avoid_node(node_name, nodes_to_avoid):
                        idle_nodes += 1
            available_nodes[partition] = idle_nodes
        except Exception as e:
            print_colored(f"Error retrieving available nodes for partition {partition}: {str(e)}", Fore.RED)
            logging.error(f"Error retrieving available nodes for partition {partition}: {str(e)}")
            available_nodes[partition] = 0

    print_colored(f"Available nodes: {available_nodes}", Fore.CYAN)
    logging.info(f"Available nodes: {available_nodes}")
    return available_nodes

def log_failed_job(exp, partition, error_message):
    failed_job = {"experiment": exp, "partition": partition, "error": error_message}
    try:
        with open("failed_jobs.txt", "a") as f:
            f.write(json.dumps(failed_job) + "\n")
        logging.error(f"Failed job logged: {failed_job}")
    except Exception as e:
        print_colored(f"Error logging failed job: {str(e)}", Fore.RED)
        logging.error(f"Error logging failed job: {str(e)}")

def submit_experiment(ssh, exp, partition):
    command = "bash ~/cleanrl/combined_exp/run.sh"
    command += f" --job-partition {partition}"
    if exp["grayscale"]:
        command += " --grayscale"
    command += f" --frame-skip {exp['frame_skip']}"
    if exp["use_compression"]:
        command += f" --compression {exp['jpeg_quality']}"
    command += f" --resolution {exp['resolution_width']} {exp['resolution_height']}"
    command += " --total-timesteps 1000000"

    print_colored(f"Submitting job: {command}", Fore.YELLOW)
    logging.info(f"Submitting job with command: {command}")
    try:
        stdin, stdout, stderr = ssh.exec_command(command)
        output = stdout.read().decode().strip()
        error = stderr.read().decode().strip()

        if error:
            print_colored(f"Error submitting job: {error}", Fore.RED)
            logging.error(f"Error submitting job: {error}")
            return None

        # Parse the job ID from the "Submitted batch job" message
        job_id_match = re.search(r"Submitted batch job (\d+)", output)
        if job_id_match:
            job_id = job_id_match.group(1)
            print_colored(f"Successfully submitted job with ID: {job_id}", Fore.GREEN)
            logging.info(f"Job submitted with ID: {job_id}")
            return job_id
        else:
            print_colored(f"Failed to get job ID. Full output: {output}", Fore.RED)
            logging.error(f"Failed to get job ID. Full output: {output}")
            return None
    except Exception as e:
        print_colored(f"Exception during job submission: {str(e)}", Fore.RED)
        logging.error(f"Exception during job submission: {str(e)}")
        return None

def check_job_status(ssh, job_id):
    try:
        command = f"squeue -j {job_id} -h -o %T"
        stdin, stdout, stderr = ssh.exec_command(command)
        state = stdout.read().decode().strip()
        logging.debug(f"Job ID {job_id} state: {state}")
        # Define known states
        known_states = ["RUNNING", "PENDING", "COMPLETED", "FAILED", "CANCELLED", 
                        "TIMEOUT", "NODE_FAIL", "PREEMPTED"]
        return state if state in known_states else "UNKNOWN"
    except Exception as e:
        print_colored(f"Error checking job status for {job_id}: {str(e)}", Fore.RED)
        logging.error(f"Error checking job status for {job_id}: {str(e)}")
        return "UNKNOWN"

def get_job_submit_time(ssh, job_id):
    try:
        command = f"sacct -j {job_id} --format=JobID,Submit --noheader --parsable2"
        stdin, stdout, stderr = ssh.exec_command(command)
        output = stdout.read().decode().strip()
        lines = output.splitlines()

        submit_time_str = None
        for line in lines:
            parts = line.strip().split('|')
            if len(parts) != 2:
                continue  # Skip malformed lines
            job_id_field, submit_time_candidate = parts
            if job_id_field == str(job_id):
                submit_time_str = submit_time_candidate.strip()
                break

        if not submit_time_str:
            raise ValueError(f"No submit time found for job {job_id}")

        # Parse the timestamp
        try:
            submit_time = datetime.datetime.strptime(submit_time_str, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            # Handle cases where the timestamp might include microseconds
            submit_time = datetime.datetime.strptime(submit_time_str, "%Y-%m-%dT%H:%M:%S.%f")

        return submit_time
    except Exception as e:
        print_colored(f"Error retrieving submit time for job {job_id}: {str(e)}", Fore.RED)
        logging.error(f"Error retrieving submit time for job {job_id}: {str(e)}")
        return datetime.datetime.now()

def cancel_job(ssh, job_id):
    try:
        command = f"scancel {job_id}"
        ssh.exec_command(command)
        print_colored(f"Cancelled job {job_id} due to exceeding 24-hour limit", Fore.RED)
        logging.warning(f"Job {job_id} cancelled due to exceeding 24-hour limit")
    except Exception as e:
        print_colored(f"Error cancelling job {job_id}: {str(e)}", Fore.RED)
        logging.error(f"Error cancelling job {job_id}: {str(e)}")

def monitor_jobs(ssh, running_jobs):
    completed_jobs = []
    current_time = datetime.datetime.now()

    for job_id, exp in running_jobs.items():
        state = check_job_status(ssh, job_id)

        if state == "RUNNING":
            submit_time = get_job_submit_time(ssh, job_id)
            runtime = current_time - submit_time
            if runtime > datetime.timedelta(hours=24):
                cancel_job(ssh, job_id)
                state = "CANCELLED"
                log_failed_job(exp, exp["partition"], "Job exceeded 24-hour limit")

        if state in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED"]:
            if state != "COMPLETED":
                print_colored(f"Job {job_id} {state}. Logging to failed_jobs.txt", Fore.RED)
                log_failed_job(exp, exp["partition"], f"Job {state}")
            completed_jobs.append(job_id)

    for job_id in completed_jobs:
        del running_jobs[job_id]

    print_colored(f"Current running jobs: {len(running_jobs)}", Fore.MAGENTA)
    logging.info(f"Current running jobs: {len(running_jobs)}")

def reconnect_ssh(ssh_details, max_retries=5, delay=60):
    for attempt in range(max_retries):
        try:
            ssh = ssh_connect_with_retry(ssh_details)
            return ssh
        except Exception as e:
            print_colored(f"Failed to reconnect SSH (attempt {attempt + 1}/{max_retries}): {str(e)}", Fore.RED)
            logging.error(f"Failed to reconnect SSH: {str(e)}")
            if attempt < max_retries - 1:
                print_colored(f"Retrying in {delay} seconds...", Fore.YELLOW)
                time.sleep(delay)
    raise Exception("Failed to reconnect SSH after multiple attempts")

def monitor_priority_queue(ssh, priority_queue_jobs, running_jobs):
    moved_to_running = []
    for job_id, exp in list(priority_queue_jobs.items()):
        try:
            state = check_job_status(ssh, job_id)
            logging.debug(f"Monitoring Priority Queue - Job ID: {job_id}, State: {state}")
            
            if state == "RUNNING":
                print_colored(f"Priority queue job {job_id} has started running", Fore.GREEN)
                logging.info(f"Priority queue job {job_id} has started running")
                running_jobs[job_id] = exp
                moved_to_running.append(job_id)
            elif state in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED"]:
                print_colored(f"Priority queue job {job_id} has {state}", Fore.YELLOW)
                logging.info(f"Priority queue job {job_id} has {state}")
                if state != "COMPLETED":
                    log_failed_job(exp, exp["partition"], f"Job {state}")
                moved_to_running.append(job_id)
            elif state == "PENDING":
                # Added detailed logging for PENDING state
                print_colored(f"Priority queue job {job_id} is still pending.", Fore.CYAN)
                logging.info(f"Priority queue job {job_id} is still pending.")
            else:
                print_colored(f"Priority queue job {job_id} is in an unknown state: {state}", Fore.RED)
                logging.warning(f"Priority queue job {job_id} is in an unknown state: {state}")
        
        except Exception as e:
            print_colored(f"Error monitoring priority queue job {job_id}: {str(e)}", Fore.RED)
            logging.error(f"Error monitoring priority queue job {job_id}: {str(e)}")
    
    for job_id in moved_to_running:
        del priority_queue_jobs[job_id]
    
    print_colored(f"Priority queue jobs after monitoring: {len(priority_queue_jobs)}", Fore.MAGENTA)
    logging.info(f"Priority queue jobs after monitoring: {len(priority_queue_jobs)}")
    
    return len(moved_to_running)

def run_experiment_loop(ssh, experiments, info):
    experiment_index = info.get("last_experiment_index", 0)
    running_jobs = {}
    nodes_to_avoid = info.get("nodes_to_avoid", [])
    priority_queue_jobs = {}
    
    # Reset priority_queue_jobs at the start
    priority_queue_jobs.clear()
    
    while experiment_index < len(experiments) or running_jobs or priority_queue_jobs:
        try:
            # Always load the latest info
            info = load_info()
            max_concurrent_experiments = info.get("max_concurrent_experiments", 10)
            max_priority_queue = info.get("max_priority_queue", 5)
            nodes_to_avoid = info.get("nodes_to_avoid", [])

            total_active_jobs = len(running_jobs) + len(priority_queue_jobs)
            available_slots = max_concurrent_experiments - total_active_jobs
            priority_queue_slots = max_priority_queue - len(priority_queue_jobs)

            print_colored("\n--- Current status ---", Fore.YELLOW, Style.BRIGHT)
            print_colored(f"Experiments submitted: {experiment_index}/{len(experiments)}", Fore.CYAN)
            print_colored(f"Running experiments: {len(running_jobs)}", Fore.CYAN)
            print_colored(f"Priority queue jobs: {len(priority_queue_jobs)}", Fore.CYAN)
            print_colored(f"Total active jobs: {total_active_jobs}", Fore.CYAN)
            print_colored(f"Available slots: {available_slots}", Fore.CYAN)
            print_colored(f"Available priority queue slots: {priority_queue_slots}", Fore.CYAN)

            if available_slots > 0 and experiment_index < len(experiments):
                available_nodes = get_available_nodes(ssh, nodes_to_avoid)

                for partition in ["bigbatch", "stampede"]:
                    partition_available_slots = available_slots
                    total_available_nodes = available_nodes.get(partition, 0)
                    if total_available_nodes > 0 or priority_queue_slots > 0:
                        jobs_to_submit = min(
                            partition_available_slots,
                            len(experiments) - experiment_index,
                            total_available_nodes + priority_queue_slots
                        )
                        for _ in range(jobs_to_submit):
                            exp = experiments[experiment_index]
                            exp["partition"] = partition
                            job_id = submit_experiment(ssh, exp, partition)
                            if job_id:
                                if total_available_nodes > 0:
                                    running_jobs[job_id] = exp
                                    total_available_nodes -= 1
                                    available_nodes[partition] = total_available_nodes
                                    print_colored(f"Experiment {experiment_index + 1} submitted with job ID: {job_id}", Fore.GREEN)
                                else:
                                    priority_queue_jobs[job_id] = exp
                                    priority_queue_slots -= 1
                                    print_colored(f"Experiment {experiment_index + 1} submitted to priority queue with job ID: {job_id}", Fore.YELLOW)
                                experiment_index += 1
                                available_slots -= 1
                                # Save the updated experiment_index
                                info["last_experiment_index"] = experiment_index
                                save_info(info)
                            else:
                                print_colored(f"Failed to submit experiment {experiment_index + 1}. Skipping...", Fore.RED)
                                logging.error(f"Failed to submit experiment {experiment_index + 1}. Skipping...")
                                experiment_index += 1
                                info["last_experiment_index"] = experiment_index
                                save_info(info)

                            if available_slots == 0 or experiment_index >= len(experiments):
                                break

                    if available_slots > 0 and experiment_index < len(experiments):
                        print_colored("No more available nodes. Waiting for 1 minute before retrying...", Fore.YELLOW)
                        time.sleep(60)  # Wait for 1 minute
                        continue

            # Monitor running and priority queue jobs
            print_colored("\nMonitoring running jobs...", Fore.YELLOW)
            monitor_jobs(ssh, running_jobs)
            print_colored("\nMonitoring priority queue jobs...", Fore.YELLOW)
            moved_jobs = monitor_priority_queue(ssh, priority_queue_jobs, running_jobs)

            # Recalculate available_slots and priority_queue_slots after monitoring
            total_active_jobs = len(running_jobs) + len(priority_queue_jobs)
            available_slots = max_concurrent_experiments - total_active_jobs
            priority_queue_slots = max_priority_queue - len(priority_queue_jobs)

            # Revalidate priority queue jobs to ensure none are stale
            for job_id in list(priority_queue_jobs.keys()):
                state = check_job_status(ssh, job_id)
                if state == "COMPLETED" or state in ["FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED"]:
                    del priority_queue_jobs[job_id]
                    print_colored(f"Revalidated and removed job {job_id} from priority queue.", Fore.GREEN)
                    logging.info(f"Revalidated and removed job {job_id} from priority queue.")

            print_colored("\n--- Updated status after monitoring ---", Fore.YELLOW, Style.BRIGHT)
            print_colored(f"Running experiments: {len(running_jobs)}", Fore.CYAN)
            print_colored(f"Priority queue jobs: {len(priority_queue_jobs)}", Fore.CYAN)
            print_colored(f"Total active jobs: {total_active_jobs}", Fore.CYAN)
            print_colored(f"Jobs moved from priority queue to running: {moved_jobs}", Fore.CYAN)
            print_colored(f"Available slots: {available_slots}", Fore.CYAN)
            print_colored(f"Available priority queue slots: {priority_queue_slots}", Fore.CYAN)

            print_colored("Waiting for 1 minute before next check...", Fore.YELLOW)
            time.sleep(60)  # Wait for 1 minute before checking again

        except (paramiko.SSHException, socket.error) as e:
            print_colored(f"SSH connection lost: {str(e)}", Fore.RED)
            logging.error(f"SSH connection lost: {str(e)}")
            print_colored("Attempting to reconnect...", Fore.YELLOW)
            ssh = reconnect_ssh(ssh_details)
        except Exception as e:
            print_colored(f"An unexpected error occurred: {str(e)}", Fore.RED)
            logging.error(f"An unexpected error occurred: {str(e)}")
            time.sleep(60)  # Wait for 1 minute before retrying

    return ssh

def load_info():
    try:
        with open("info.json", "r") as f:
            info = json.load(f)
        return info
    except Exception as e:
        print_colored(f"Error loading info.json: {str(e)}", Fore.RED)
        logging.error(f"Error loading info.json: {str(e)}")
        # Return default configuration if loading fails
        return {
            "last_experiment_index": 0,
            "max_concurrent_experiments": 10,
            "max_priority_queue": 5,
            "nodes_to_avoid": []
        }

def save_info(info):
    try:
        with open("info.json", "w") as f:
            json.dump(info, f, indent=2)
        logging.info(f"Info saved: {info}")
    except Exception as e:
        print_colored(f"Error saving info.json: {str(e)}", Fore.RED)
        logging.error(f"Error saving info.json: {str(e)}")

def main():
    info = load_info()
    experiments = load_experiments("all_experiments.txt")
    ssh = ssh_connect_with_retry(ssh_details)

    try:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                ssh = run_experiment_loop(ssh, experiments, info)
                break  # If successful, exit the retry loop
            except Exception as e:
                print_colored(f"An error occurred (attempt {attempt + 1}/{max_retries}): {str(e)}", Fore.RED)
                logging.error(f"An error occurred: {str(e)}")
                if attempt < max_retries - 1:
                    print_colored("Retrying in 5 minutes...", Fore.YELLOW)
                    time.sleep(300)  # Wait for 5 minutes before retrying
                else:
                    print_colored("Max retries reached. Exiting.", Fore.RED)
                    raise
    except Exception as e:
        print_colored(f"Fatal error: {str(e)}", Fore.RED)
        logging.critical(f"Fatal error: {str(e)}")
    finally:
        ssh.close()
        print_colored("\nAll experiments have been submitted and completed!", Fore.GREEN, Style.BRIGHT)
        logging.info("All experiments have been submitted and completed!")

if __name__ == "__main__":
    main()
