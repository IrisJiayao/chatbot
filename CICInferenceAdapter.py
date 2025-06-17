# -*- coding: utf-8 -*-
# CICInferenceAdapter.py

import os
import time
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from config.config import Config


class CICAdapter:
    """
    Adapter class to interface with the CIC-IDS2018 inference model
    """

    def __init__(self, config_path):
        """
        Initialize the CIC adapter with configuration

        Args:
            config_path (str): Path to the configuration file
        """
        try:
            self.config_path = config_path
            self.config = Config(conf_file_path=config_path, exp_name="CSE-CIC-IDS2018")

            # Default model path, can be updated later - using forward slashes to avoid escape issues
            self.model_path = "D:/PythonProject/chatbot/log/CSE-CIC-IDS2018_1s/03-26-2025-06-21-08/epoch_12.pth"

            # Default test data path
            self.test_path = "D:/PythonProject/chatbot/dataset/preprocessed/CSE-CIC-IDS2018_1s/test.csv"

            # Output directory for graphs
            self.output_dir = Path("./output/predictions")
            self.output_dir.mkdir(exist_ok=True, parents=True)

            self._inference = None

            print("✅ CIC adapter initialized successfully")
        except Exception as e:
            print(f"⚠️ Error initializing CIC adapter: {str(e)}")
            raise

    def _get_inference_engine(self):

        if self._inference is None:
            try:
                from infer import Inference
                self._inference = Inference(
                    self.config,
                    test_path=self.test_path,
                    model_path=self.model_path
                )
            except Exception as e:
                print(f"⚠️ Error initializing inference engine: {str(e)}")
                raise
        return self._inference

    def predict(self, analysis_type, scope, target_ip=None, time_point=None, is_multistep=False, use_iterative=False,
                prediction_steps=10):
        """
        Make a prediction based on the requested analysis type

        Args:
            analysis_type (str): Type of analysis - 'overall' or 'timeslot'
            scope (str): Scope of analysis - always 'network' for CIC
            target_ip (str, optional): Not used for CIC
            time_point (str, optional): Time point for timeslot analysis (YYYY-MM-DD-HH:MM)
            is_multistep (bool): Whether to use multi-step prediction
            use_iterative (bool): Whether to use iterative multi-step prediction
            prediction_steps (int): Number of steps for prediction (for iterative method)

        Returns:
            dict: Prediction result
        """
        # Validate parameters
        if scope != "network":
            return {
                "success": False,
                "message": "CIC-IDS2018 only supports network-wide analysis"
            }

        if analysis_type not in ["overall", "timeslot"]:
            return {
                "success": False,
                "message": f"Unsupported analysis type: {analysis_type}"
            }

        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_subdir = self.output_dir / timestamp
            output_subdir.mkdir(exist_ok=True, parents=True)

            # 获取推理引擎
            inference = self._get_inference_engine()

            # Run the appropriate prediction
            if analysis_type == "overall":
                if is_multistep:
                    if use_iterative:
                        # The iterative method is used for multi-step prediction
                        result = inference.run_iterative_multistep(
                            prediction_steps=prediction_steps,
                            save_plot=True,
                            output_dir=output_subdir,
                            show_plot=False,
                            silent=True
                        )
                    else:
                        result = inference.run_multistep_inference(
                            prediction_steps=prediction_steps,
                            save_plot=True,
                            output_dir=output_subdir,
                            show_plot=False,
                            silent=True
                        )
                else:
                    result = inference.run_inference(
                        save_plot=True,
                        output_dir=output_subdir,
                        show_plot=False,
                        silent=True
                    )

                # Prepare additional details for the response
                result["details"] = {
                    "scope": "network-wide",
                    "analysis_type": "overall",
                    "prediction_steps": prediction_steps if is_multistep else 1
                }

                if is_multistep:
                    method_type = "iterative multi-step" if use_iterative else "multi-step"
                    result[
                        "message"] = f"CIC-IDS2018 overall {method_type} prediction completed ({prediction_steps} steps)"
                else:
                    result["message"] = "CIC-IDS2018 overall single-step prediction completed"

            elif analysis_type == "timeslot":
                if not time_point:
                    return {
                        "success": False,
                        "message": "Time point is required for timeslot analysis"
                    }

                result = inference.predict_timeslot(
                    time_point=time_point,
                    is_multistep=is_multistep,  # 直接传递多步标志
                    save_plot=True,
                    output_dir=output_subdir,
                    show_plot=False,
                    silent=True
                )

                result["details"] = {
                    "scope": "network-wide",
                    "analysis_type": "timeslot",
                    "time_point": time_point,
                    "prediction_steps": prediction_steps if is_multistep else 1
                }

                if is_multistep:
                    result["message"] = f"CIC-IDS2018 timeslot multi-step prediction for {time_point} completed"
                else:
                    result["message"] = f"CIC-IDS2018 timeslot single-step prediction for {time_point} completed"

            return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "message": f"Error making prediction: {str(e)}"
            }

    def predict_with_iterative_multistep(self, analysis_type, time_point=None, prediction_steps=10, save_plot=True,
                                         output_dir=None):
        """
        The prediction is made using iterative multi-step prediction

        Parameters:
        analysis_type (str): Analysis type - 'overall' or 'timeslot'
        time_point (str, optional): Time point in time format: YYYY-MM-DD-HH:MM
        prediction_steps (int): indicates the number of predicted steps
        save_plot (bool): Whether to save graphs
        output_dir (str, optional): graphics save directory

        Back:
        dict: indicates the prediction result
        """
        return self.predict(
            analysis_type=analysis_type,
            scope="network",
            target_ip=None,
            time_point=time_point,
            is_multistep=True,
            use_iterative=True,
            prediction_steps=prediction_steps
        )


# Example usage
if __name__ == "__main__":
    # Using forward slashes to avoid escape sequence warnings
    config_path = "D:/PythonProject/chatbot/config/config/CSE-CIC-IDS2018.yaml"
    try:
        adapter = CICAdapter(config_path)

        print("\nTesting timeslot prediction (single-step):")
        result = adapter.predict("timeslot", "network", time_point="2018-03-02-12:12", is_multistep=False)
        if result["success"]:
            print(f"Prediction for 2025-03-28-10:00: {result['prediction']:.2f} Mbps")
            print(f"Accuracy: {result['accuracy']:.2f}%")
            if result.get("graphs"):
                print(f"Generated graph: {result['graphs'][0]}")
        else:
            print(f"Error: {result['message']}")

        # Test timeslot prediction (multi-step)
        print("\nTesting timeslot prediction (multi-step):")
        result = adapter.predict("timeslot", "network", time_point="2025-03-28-10:00", is_multistep=True)
        if result["success"]:
            print(f"Final step prediction for 2025-03-28-10:00+: {result['prediction']:.2f} Mbps")
            print(f"Final step accuracy: {result['accuracy']:.2f}%")
            if result.get("graphs"):
                print(f"Generated graph: {result['graphs'][0]}")
        else:
            print(f"Error: {result['message']}")

    except Exception as e:
        print(f"Error in example: {e}")