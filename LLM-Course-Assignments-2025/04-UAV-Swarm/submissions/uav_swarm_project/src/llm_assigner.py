import json
from openai import OpenAI


class LLMUAVTaskAssigner:
    """LLM任务分配类：适配硅基流动deepseek3模型，增加容错和调试"""

    def __init__(self, api_key, base_url, model="deepseek3"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        print(f"LLM分配器初始化完成（硅基流动），使用模型：{self.model}")

    def build_prompt(self, uav_pos_dict, patrol_points):
        """优化Prompt，适配deepseek3的格式约束（更强制、更简洁）"""
        prompt = f"""
        指令：
        1. 仅输出JSON格式的无人机巡检点分配结果，无任何前置、后置文字，无解释、无说明；
        2. JSON必须包含UAV_0、UAV_1、UAV_2、UAV_3四个键，值为对应的巡检点坐标数组；
        3. 每个巡检点仅分配给1架无人机，不重复、不遗漏；
        4. 优先分配距离无人机最近的巡检点。

        无人机当前坐标：{json.dumps(uav_pos_dict)}
        所有巡检点坐标：{json.dumps(patrol_points)}

        输出示例（仅参考格式，需按实际分配）：
        {{"UAV_0":[10,2,-5],"UAV_1":[8,5,-5],"UAV_2":[12,8,-5],"UAV_3":[15,3,-5]}}
        """
        return prompt.strip()

    def extract_json_from_output(self, llm_output):
        """容错处理：从LLM输出中提取JSON部分（适配deepseek3的多余文字）"""
        if not llm_output:
            return None

        # 去除首尾空格/换行
        llm_output = llm_output.strip()

        # 情况1：输出直接是JSON（理想情况）
        try:
            return json.loads(llm_output)
        except:
            pass

        # 情况2：输出包含JSON片段（如“结果：{...}”），提取{}之间的内容
        import re
        json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                return json.loads(json_str)
            except:
                print(f"❌ 提取的JSON片段解析失败：{json_str}")
                return None

        # 情况3：无有效JSON
        return None

    def assign_patrol_points(self, uav_pos_dict, patrol_points):
        """调用deepseek3，增加详细调试和容错"""
        prompt = self.build_prompt(uav_pos_dict, patrol_points)
        try:
            # 调用API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                timeout=30
            )

            # 1. 获取原始输出并打印（核心调试）
            llm_output = response.choices[0].message.content
            print(f"📝 deepseek3原始输出（完整）：[{llm_output}]")  # 用[]包裹，便于看空内容

            # 2. 容错提取JSON
            assign_result = self.extract_json_from_output(llm_output)
            if not assign_result:
                raise ValueError(f"无法从LLM输出中提取有效JSON，原始输出：{llm_output}")

            # 3. 验证结果合法性
            required_uavs = list(uav_pos_dict.keys())
            if not all(uav in assign_result for uav in required_uavs):
                raise ValueError(f"JSON缺少无人机键，仅包含：{list(assign_result.keys())}，要求：{required_uavs}")

            print(f"✅ LLM任务分配完成：{assign_result}")
            return assign_result

        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败：{e}，LLM原始输出：{llm_output if 'llm_output' in locals() else '空'}")
            return None
        except ValueError as e:
            print(f"❌ 分配结果验证失败：{e}")
            return None
        except Exception as e:
            print(f"❌ LLM调用/解析失败：{e}")
            return None