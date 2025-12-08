"""
LLM Agent Module

Implements the LLM-based agent for autonomous simulation refinement.
Handles communication with various LLM APIs and prompt engineering.
"""
import re
import os
from dotenv import load_dotenv
from openai import OpenAI
from src.refinement.files_operation import load_prompts, load_env_cfg

class EurekaAgent():
    def __init__(self, task_config: dict, agent_config: dict):
        # task_config = {
        #     "workspace": workspace,
        #     "env_cfg_path": os.path.join(workspace, task_yaml.get('env_cfg_path')),
        #     "logs_path": os.path.join(workspace, task_yaml.get('logs_path')),
        #     "task_description": task_yaml.get('description'),
        # }
        self.prompts_dict = load_prompts()
        self.task_description = task_config["task_description"]
        self.env_cfg_dict = load_env_cfg(task_config["env_cfg_path"])
        self.model = agent_config.get('model')
        self.base_url = agent_config.get('base_url')
        self.samples = agent_config.get('sample', 4)
        # record
        self.refine_record = []
        self.best_idx = []
        # load_dotenv()
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        # init messages
        system_content, user_content = self.init_prompt()
        self.messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]


    def init_prompt(self):
        # system prompt
        self.code_output_tip = self.prompts_dict["code_output_tip"]
        reward_template = self.env_cfg_dict["reward_code"]
        initial_system = self.prompts_dict["initial_system"]
        # feedback
        self.code_feedback = self.prompts_dict["code_feedback"]
        self.execution_error_feedback = self.prompts_dict["execution_error_feedback"]
        self.policy_feedback = self.prompts_dict["policy_feedback"]

        system_content = initial_system.format(task_reward_template=reward_template) + self.code_output_tip
        # user prompt
        # task_dict = {
        #     "code_output_tip": code_output_tip,
        #     "initial_system": initial_system,
        #     "initial_user": initial_user,
        #     "reward_signature": reward_signature,
        #     "observation_code": observation_code,
        #     "reward_code": reward_code,
        #     "intermediate_code": intermediate_code,
        #     "env_cfg_code": env_cfg_code,
        #     "env_file_path": env_cfg_path
        # }
        initial_user = self.prompts_dict["initial_user"]
        # task_obs_code_string = (
        #     self.env_cfg_dict["observation_code"]
        #     + self.env_cfg_dict["intermediate_code"]
        # )
        task_obs_code_string = self.env_cfg_dict["env_code"]
        user_content= initial_user.format(task_obs_code_string=task_obs_code_string, task_description=self.task_description)
        return system_content, user_content


    def receive_feedback(self, refine_logs: dict, iteration: int):
        # refine_record[i].append({
        #     'ckpt': log_path,
        #     'max_con_successes': max_con_successes,
        #     'tb_path': tb_path,
        #     'prompt': llm_agent.messages,
        #     'reward_func': reward_func,
        #     "responses_content": raw_response,
        #     "feedback_path": os.path.join(log_path, "training_record", "training_summary.txt")
        # })
        if refine_logs is None:
            feedback_content = self.execution_error_feedback(traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")

        else:
            # TODO: seed
            best_samples = refine_logs[f'iteration_{iteration}/eval'][0]["idx"]
            eval_logs = refine_logs[f'iteration_{iteration}/eval'][1]
            best_eval_idx = next((i for i, log in enumerate(eval_logs) if log.get("samples") == best_samples), None)

            feedback_content = self.policy_feedback
            with open(os.path.join(eval_logs[best_eval_idx]["result"]["log_path"], "training_record", "training_summary.txt"), "r") as f:
                feedback_content += f.read()
            feedback_content += self.code_feedback

        feedback_content += self.code_output_tip

        # Add feedback message to the conversation history
        best_run_idx = refine_logs[f'iteration_{iteration}/run'][0]['idx']
        if len(self.messages) == 2:
            self.messages += [{"role": "assistant", "content": refine_logs[f'iteration_{iteration}/run'][3][best_run_idx]}]
            self.messages += [{"role": "user", "content": feedback_content}]
        else:
            assert len(self.messages) == 4
            self.messages[-2] = {"role": "assistant", "content": refine_logs[f'iteration_{iteration}/run'][3][best_run_idx]}
            self.messages[-1] = {"role": "user", "content": feedback_content}
    

    def func_gen(self, messages):
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # sending the messaages to the LLM API and get the response
                completion = self.client.chat.completions.create(
                    extra_headers={},
                    extra_body={},
                    model=self.model,
                    temperature=0.8,
                    n=1,
                    messages=messages
                )
                response = completion.choices[0].message.content
                # filtering the response to extract only the python function code
                patterns = [
                    r'```python(.*?)```',
                    r'```(.*?)```',
                    r'"""(.*?)"""',
                    r'""(.*?)""',
                    r'"(.*?)"',
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, response, re.DOTALL)
                    if matches:
                        code_blocks = matches[0].strip()
                        if re.search(r'@torch\.jit\.script\s*\n*def\s+compute_rewards\s*\([^)]*\).*?return\s+total_reward, reward_components', code_blocks, re.DOTALL):
                            return code_blocks, response
                
                # If we get here, no valid code block was found
                retry_count += 1
            except Exception as e:
                retry_count += 1
                print(f"Attempt {retry_count} failed: {str(e)}")
            
            # If we've exhausted all retries
        raise RuntimeError("Failed to generate valid reward function code after 10 attempts.")
    

    # def one_move(self, env_evaluation):
    #     self.refine_record.append(list())
    #     i=0
    #     while i < self.samples:
    #         # generate a new reward function string
    #         reward_func, raw_response = self.func_gen(self.messages)
    #         print("The Reward func generated...")
    #         # test the generated reward function in the environment
    #         train_result = env_evaluation(reward_func, log_name=f"iter_{len(self.refine_record)}_sample{i}")
    #         print("evaluated!!")
    #         # record it
    #         if train_result is not None:
    #             self.refine_record[-1].append({
    #                 "train_result": train_result,
    #                 "prompt": self.messages,
    #                 "reward_func": reward_func,
    #                 "responses_content": raw_response,
    #                 "feedback_path": os.path.join(train_result["log_path"], "training_record", "training_summary.txt")
    #             })
    #             i += 1
    #         print("valid recorded!!")
    #     # Finding the best performance among samples
    #     best_idx = max(range(self.samples), key=lambda idx: self.refine_record[-1][idx]['train_result']['max_con_successes'])
    #     self.best_idx.append(best_idx)
    #     # OPTIONAL: Print the best performance
    #     print(f"Best performance (sample {best_idx}): {self.refine_record[-1][best_idx]}")
    #     return reward_func
