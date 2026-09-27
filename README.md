# 输出增强

![:name](https://count.getloli.com/@astrbot_plugin_output_enhance?name=astrbot_plugin_output_enhance&theme=miku&padding=7&offset=0&align=top&scale=1&pixelated=1&darkmode=auto)

AstrBot 插件：让模型用输出标记表达 @、引用、分段和转图片，并在发送前自动处理长文本。

## 使用前需要关闭的 AstrBot 设置

插件不会修改这些设置。对应能力开启时如果检测到冲突，只会在日志里提醒。

| 使用的插件能力 | 需要关闭的 AstrBot 设置 |
| --- | --- |
| 分段回复 | `platform_settings.segmented_reply.enable` |
| 合并转发 | 将 `platform_settings.forward_threshold` 调到足够大，避免框架先折叠 |
| 文转图片 | `t2i` |
| 输出标记解析 | `provider_settings.streaming_response`（流式输出会跳过发送前装饰，标记无法解析） |
| 自动引用 | `platform_settings.reply_with_quote`（仅在不希望框架自动引用时关闭） |

## 输出标记

标记写在回复正文里，发送前替换成真实消息组件。

| 标记 | 作用 |
| --- | --- |
| `{{AT:用户ID}}` | @ 指定用户，只接受数字 ID |
| `{{AT:all}}` | @全体成员，需同时允许 |
| `{{REPLY}}` | 引用触发这条回复的消息 |
| `{{SEG}}` | 在这里拆成两条消息 |
| `{{IMG}}` | 把整条回复渲染成图片 |

无法识别、未闭合，或对应功能未开启的标记会被移除。当前平台不支持 @、引用或合并转发时，相关处理会被跳过，其余内容照常发送。

## 主动工具

- `send_forward_message`：按模型给出的节点发送合并转发。
- `send_text_as_image`：把文本渲染成图片发送。只接收文本，不接收图片路径或 URL。

未在 `llm_tool_options` 勾选的工具不会注册。

## 被动处理

发送前依次执行文本清洗、分段、阈值判断。达到文转图片阈值时渲染成图片；达到合并转发阈值时，仅在支持的平台折叠成合并转发。按句数折叠时，每个分段是转发里的一条消息；只按字数折叠、没有分段时仍是一条。图片阈值同时达到时优先转图片。处理失败会保留原文并记录日志。
