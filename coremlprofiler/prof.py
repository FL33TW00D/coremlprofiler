import os
from pathlib import Path
from Foundation import NSURL
from CoreML import MLModel, MLModelConfiguration, MLComputePlan
from PyObjCTools import AppHelper
import enum
from colorama import Fore, Style


class ComputeDevice(enum.Enum):
    CPU = 0
    GPU = 1
    ANE = 2
    Unknown = 3

    def __str__(self):
        return self.name

    @classmethod
    def from_pyobjc(cls, device):
        from CoreML import (
            MLCPUComputeDevice,
            MLGPUComputeDevice,
            MLNeuralEngineComputeDevice,
        )

        if isinstance(device, MLCPUComputeDevice):
            return cls.CPU
        elif isinstance(device, MLGPUComputeDevice):
            return cls.GPU
        elif isinstance(device, MLNeuralEngineComputeDevice):
            return cls.ANE
        else:
            return cls.Unknown


class DeviceUsage(dict):
    def __init__(self):
        super().__init__(
            {
                ComputeDevice.CPU: 0,
                ComputeDevice.GPU: 0,
                ComputeDevice.ANE: 0,
            }
        )

    def __str__(self):
        return ", ".join(f"{device}: {count}" for device, count in self.items())


class CoreMLProfiler:
    def __init__(self, model_path: str):
        self.model_url = self._validate_and_prepare_model(model_path)
        self.compute_plan = None
        self.device_usage = None

    def _validate_and_prepare_model(self, model_path: str) -> NSURL:
        """Validate the model path and convert if necessary."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"File {model_path} does not exist")

        compiled_path = None
        if model_path.endswith(".mlpackage"):
            if os.path.exists(model_path.replace(".mlpackage", ".mlmodelc")):
                compiled_path = model_path.replace(".mlpackage", ".mlmodelc")
            else:
                compiled_path = self._convert_mlpackage_to_mlmodelc(model_path)
        elif model_path.endswith(".mlmodelc"):
            compiled_path = model_path
        else:
            raise ValueError("Input file must be either .mlpackage or .mlmodelc")

        return NSURL.fileURLWithPath_(compiled_path)

    def _convert_mlpackage_to_mlmodelc(self, input_path) -> Path:
        """Convert .mlpackage to .mlmodelc."""
        compiled_path, error = MLModel.compileModelAtURL_error_(
            NSURL.fileURLWithPath_(input_path), None
        )
        if error:
            raise ValueError(f"Error compiling model: {error}")
        output_path = Path(input_path).with_suffix(".mlmodelc")
        Path(compiled_path).rename(output_path)
        return str(output_path)

    def _create_compute_plan(self):
        """Create a compute plan for the model."""
        config = MLModelConfiguration.alloc().init()
        MLComputePlan.loadContentsOfURL_configuration_completionHandler_(
            self.model_url, config, self._handle_compute_plan
        )
        AppHelper.runConsoleEventLoop(installInterrupt=True)

    def _handle_compute_plan(self, compute_plan, error):
        """Handle the compute plan callback."""
        if error:
            raise RuntimeError(f"Error loading compute plan: {error}")

        if compute_plan:
            self.compute_plan = compute_plan
        else:
            raise ValueError("No compute plan returned")

        AppHelper.callAfter(AppHelper.stopEventLoop)

    def _calculate_device_usage(self) -> DeviceUsage:
        if not self.compute_plan:
            self._create_compute_plan()

        program = self.compute_plan.modelStructure().program()
        if not program:
            raise ValueError("Missing program")

        main_function = program.functions().objectForKey_("main")
        if not main_function:
            raise ValueError("Missing main function")

        operations = main_function.block().operations()

        self.device_usage = DeviceUsage()
        for operation in operations:
            device_usage = self.compute_plan.computeDeviceUsageForMLProgramOperation_(
                operation
            )
            if device_usage:
                device_type = ComputeDevice.from_pyobjc(
                    device_usage.preferredComputeDevice()
                )
                self.device_usage[device_type] += 1

        return self.device_usage

    def device_usage_summary(self) -> DeviceUsage:
        """Return a summary of device usage."""
        if not self.device_usage:
            self._calculate_device_usage()
        return self.device_usage

    def device_usage_summary_chart(self, total_width=50):
        """Create a bar chart representation of device counts similar to XCode."""
        if not self.device_usage:
            self._calculate_device_usage()
        total = sum(self.device_usage.values())
        title = "Compute Unit Mapping"
        bar = ""
        legend = f"All: {total}  "
        colors = {
            ComputeDevice.CPU: Fore.BLUE,
            ComputeDevice.GPU: Fore.GREEN,
            ComputeDevice.ANE: Fore.MAGENTA,
            ComputeDevice.Unknown: Fore.YELLOW,
        }

        for device, count in self.device_usage.items():
            width = int(count / total * total_width) if total > 0 else 0
            bar += colors[device] + "█" * width
            legend += f"{colors[device]}■{Style.RESET_ALL} {device}: {count}  "

        return f"\033[1m{title}\033[0m\n{bar}{Style.RESET_ALL}\n{legend}"
