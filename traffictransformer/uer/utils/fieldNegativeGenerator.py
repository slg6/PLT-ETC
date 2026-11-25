import random
from copy import deepcopy

class FieldNegativeGenerator:
    def __init__(self):
        self.ip_fields = [
            "time_delta",
            "direction",
            # "version",
            # "headlen",
            "tos",
            "len",
            # "id",
            "flags",
            # "frag",
            "ttl",
            "proto"
        ]
        self.tcp_fields = [
            # "sport",
            # "dport",
            # "seq_low",
            # "ack_low",
            "dataofs",
            # "res",
            "flags",
            "window"
        ]
        self.udp_fields = [
            # "sport",
            # "dport",
            "len",
            "reserved1",
            "reserved2",
            # "reserved3",
            # "reserved4",
            # "reserved5"
        ]
        self.total_fields = len(self.ip_fields) + max(len(self.tcp_fields), len(self.udp_fields))
        self.constraints = {
            "ip": {
                # "time_delta": {"type": "range", "min": "0000", "max": "1388"},  # 0-5000ms（0x1388=5000）
                "direction": {"type": "direction"},
                # "version": {"type": "enum", "values": ["0004"]},  # IPv4/IPv6
                "tos": {"type": "range", "min": "0000", "max": "00ff"},
                "len": {"type": "range", "min": "0014", "max": "ffff"},
                "flags": {"type": "enum", "values": ["0000", "0001", "0002", "0003"]},
                "ttl": {"type": "range", "min": "0001", "max": "00ff"},  # 1-255
                "proto": {"type": "enum", "values": ["0006", "0011"]}  # TCP=6, UDP=17
            },
            "tcp": {
                # "sport": {"type": "fixed_in_flow"},
                # "dport": {"type": "fixed_in_flow"},
                "dataofs": {"type": "range", "min": "0005", "max": "000f"},
                "flags": {"type": "enum", "values": [
                    "0002",
                    "0004",
                    "0010",
                    "0012",  # SYN+ACK  1
                    "0011",  # FIN+ACK  1
                    "0018",  # PSH+ACK 1
                    "0014",  # RST+ACK 1
                ]},
                "window": {"type": "range", "min": "0001", "max": "ffff"},
                # "urgptr": {"type": "range", "min": "0000", "max": "ffff"}
            },
            "udp": {
                # "sport": {"type": "fixed_in_flow"},
                # "dport": {"type": "fixed_in_flow"},
                "len": {"type": "range", "min": "0008", "max": "ffeb"},
                "reserved1": {"type": "fixed", "value": "0000"},
                "reserved2": {"type": "fixed", "value": "0000"},
                # "reserved3": {"type": "fixed", "value": "0000"},
                # "reserved4": {"type": "fixed", "value": "0000"},
                # "reserved5": {"type": "fixed", "value": "0000"},
            }
        }

    def generate_negative_sample(self, modified, protocol_type,  modify_count, field_mask_list, packet_idex):
        """
        生成负样本并标记修改位置
        :param original_fields: 原始字段列表（ip字段+tcp/udp字段）
        :param protocol_type: 协议类型（"tcp"/"udp"）
        :param flow_context: 流上下文（含历史端口信息），格式：
            {
                "flow_id": "xxx",
                "history_sports": ["dd02", ...],  # 该流历史源端口
                "history_dports": ["01bb", ...]   # 该流历史目的端口
            }
        :param modify_count: 修改字段数量
        :return: (修改后字段列表, 修改标记列表)
        """
        protocol_type = protocol_type.lower()
        if protocol_type.lower() not in ["tcp", "udp"]:
            raise ValueError("协议类型必须为'tcp'或'udp'")
        if len(modified) != self.total_fields:
            raise ValueError(f"字段数量错误，应为{self.total_fields}个")
        # modified = deepcopy(original_fields)
        proto_fields = self.tcp_fields if protocol_type.lower() == "tcp" else self.udp_fields

        all_indices = list(range(self.total_fields))
        selected_indices = random.sample(all_indices, min(modify_count, len(all_indices)))
        # driection = modified[self.ip_fields.index("direction")]
        for idx in selected_indices:
            if idx < len(self.ip_fields):
                field_type = "ip"
                field_name = self.ip_fields[idx]
            else:
                field_type = protocol_type
                field_name = proto_fields[idx - len(self.ip_fields)]

            current_val = modified[idx]
            if field_name not in self.constraints[field_type]:
                continue
            if field_name == "window" and current_val == 5:
                continue
            constraints = self.constraints[field_type][field_name]

            # if current_val == 0:
            invalid_val = self._get_invalid_value(current_val, field_name, constraints)
            src_index = idx + 1 + packet_idex*(self.total_fields+1)
            field_mask_list.append((src_index, modified[idx]))
            modified[idx] = invalid_val


    def _get_invalid_value(self, current_val, field_name, constraints):
        """根据字段约束生成无效值
            current_val: 当前字段值（十六进制字符串）
            field_name: 字段名称（如"sport"）
            constraints: 该字段的约束字典
        """
        current_val = current_val - 5
        hex_to_int = lambda h: int(h, 16)
        int_add_five = lambda i: i + 5

        # if field_name in ["sport", "dport"] and constraints["type"] == "fixed_in_flow":
        #     while True:
        #         invalid = (random.randint(0, 65535))
        #         if invalid != current_val:
        #             return int_add_five(invalid)

        if constraints["type"] == "enum":
            while True:
                invalid = (random.randint(0, 0xffff))
                if invalid not in constraints["values"] and invalid != current_val:
                    return int_add_five(invalid)
        elif constraints["type"] == "range":
            min_val = hex_to_int(constraints["min"])
            max_val = hex_to_int(constraints["max"])
            while True:
                invalid_int = random.choice([
                    random.randint(0, min_val - 1) if min_val > 0 else None,
                    random.randint(max_val + 1, 0xffff) if max_val < 0xffff else None
                ])
                if invalid_int is not None:
                    invalid = (invalid_int)
                    if invalid != current_val:
                        return int_add_five(invalid)
        elif constraints["type"] == "fixed":
            invalid = (random.randint(0, 0xffff))
            while True:
                invalid = (random.randint(0, 0xffff))
                if invalid != current_val:
                    return int_add_five(invalid)
        # elif constraints["type"] == "increasing":
        #     if current_val == 0:
        #     else:
        elif constraints["type"] == "direction":
            return int_add_five(2 if current_val == 1 else 1)