import argparse
import os
import sys
from Foundation import *
from CoreML import *
import logging
from PyObjCTools import AppHelper
import enum
from colorama import Fore, Style
from pathlib import Path

def convert_mlpackage_to_mlmodelc(input_path):
    url = NSURL.fileURLWithPath_(input_path)
    compiled_path, error = MLModel.compileModelAtURL_error_(url, None)
    if error:
        logging.error(f"Error compiling model: {error}")
        sys.exit(1)
    output_path = Path(input_path).with_suffix(".mlmodelc")
    Path(compiled_path).rename(output_path)
    return output_path

class ComputeDevice(enum.Enum):
    CPU = 0
    GPU = 1
    ANE = 2
    Unknown = 3

    def __str__(self):
        return self.name

    @classmethod
    def from_pyobjc(cls, device):
        if isinstance(device, MLCPUComputeDevice):
            return cls.CPU
        elif isinstance(device, MLGPUComputeDevice):
            return cls.GPU
        elif isinstance(device, MLNeuralEngineComputeDevice):
            return cls.ANE
        else:
            return cls.Unknown

def create_bar_chart(device_counts, total_width=50):
    total = sum(device_counts.values())
    title = "Compute Unit Mapping"
    bar = ""
    legend = f"All: {total}  "
    colors = {
        ComputeDevice.CPU: Fore.BLUE,
        ComputeDevice.GPU: Fore.GREEN,
        ComputeDevice.ANE: Fore.MAGENTA,
        ComputeDevice.Unknown: Fore.YELLOW
    }
    
    for device, count in device_counts.items():
        width = int(count / total * total_width)
        bar += colors[device] + '█' * width
        legend += f"{colors[device]}■{Style.RESET_ALL} {device}: {count}  "

    return f"\033[1m{title}\033[0m\n{bar}{Style.RESET_ALL}\n{legend}"

def handle_compute_plan(compute_plan, error):
    if error:
        logging.error(f"Error loading compute plan: {error}")
        AppHelper.callAfter(AppHelper.stopEventLoop)
        return
    
    if compute_plan:
        program = compute_plan.modelStructure().program()
        if not program:
            logging.error("Missing program")
            return
        
        main_function = program.functions().objectForKey_("main")
        if not main_function:
            logging.error("Missing main function")
            return
        
        operations = main_function.block().operations()
        device_counts = {
            ComputeDevice.CPU: 0,
            ComputeDevice.GPU: 0,
            ComputeDevice.ANE: 0,
        }
        
        for operation in operations:
            #print(f"\nOperation: {operation.operatorName()}")
            device_usage = compute_plan.computeDeviceUsageForMLProgramOperation_(operation)
            estimated_cost = compute_plan.estimatedCostOfMLProgramOperation_(operation)
            if device_usage:
                device_type = ComputeDevice.from_pyobjc(device_usage.preferredComputeDevice())
                device_counts[device_type] += 1
            if estimated_cost:
                weight = estimated_cost.weight()

        print(create_bar_chart(device_counts))
    else:
        logging.error("No compute plan returned")
        return
    
    AppHelper.callAfter(AppHelper.stopEventLoop)

def main():
    parser = argparse.ArgumentParser(description="Analyze CoreML models (.mlpackage or .mlmodelc)")
    parser.add_argument("model_path", help="Path to the .mlpackage or .mlmodelc file")
    args = parser.parse_args()

    model_path = args.model_path
    if not os.path.exists(model_path):
        logging.error(f"Error: File {model_path} does not exist")
        sys.exit(1)

    if model_path.endswith('.mlpackage'):
        model_path = convert_mlpackage_to_mlmodelc(model_path)
    elif model_path.endswith('.mlmodelc'):
        model_path = NSURL.fileURLWithPath_(model_path) 
    else:
        logging.error("Error: Input file must be either .mlpackage or .mlmodelc")
        sys.exit(1)

    # Load the compute plan
    config = MLModelConfiguration.alloc().init()
    MLComputePlan.loadContentsOfURL_configuration_completionHandler_(
        model_path,
        config,
        handle_compute_plan
    )

    # Start the run loop
    AppHelper.runConsoleEventLoop(installInterrupt=True)

if __name__ == "__main__":
    main()
