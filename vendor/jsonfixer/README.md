# jsonfixer

一个用于修复损坏的 JSON 字符串的 Rust 库。

A Rust library for fixing malformed JSON strings.

## 功能特性 / Features

- 🛠️ **JSON 修复**：修复常见的 JSON 语法错误 / **JSON Repair**: Fix common JSON syntax errors
- 🤖 **LLM 友好**：专门设计用于处理来自 LLM 的 JSON 输出 / **LLM Friendly**: Specifically designed to handle JSON output from LLMs
- 📁 **多种输入源**：支持字符串、文件和读取器 / **Multiple Input Sources**: Support for strings, files, and readers
- 🔧 **灵活输出**：可返回修复后的 JSON 字符串或解析后的 `serde_json::Value` 对象 / **Flexible Output**: Return repaired JSON strings or parsed `serde_json::Value` objects
- 📊 **详细日志**：提供带有修复日志的全面错误报告 / **Detailed Logging**: Comprehensive error reports with repair logs
- ⚡ **CLI 工具**：用于快速文件修复的命令行界面 / **CLI Tool**: Command-line interface for quick file repairs

## 安装 / Installation

在 `Cargo.toml` 中添加 / Add to `Cargo.toml`：

```toml
[dependencies]
jsonfixer = "0.1.0"
```

## 快速开始 / Quick Start

### 基本用法 / Basic Usage

```rust
use jsonfixer::{repair_json, loads, JsonRepairOptions};

// 修复 JSON 字符串并返回字符串 / Repair JSON string and return string
let options = JsonRepairOptions::default();
let repaired = repair_json(r#"{"name": John, "age": 30}"#, options).unwrap();
// 返回: "{\"name\": \"John\", \"age\": 30}" / Returns: "{\"name\": \"John\", \"age\": 30}"

// 修复 JSON 并返回解析后的 Value / Repair JSON and return parsed Value
let parsed = loads(r#"{"name": John, "age": 30}"#, options).unwrap();
// 返回 serde_json::Value / Returns serde_json::Value

// 从文件修复 JSON / Repair JSON from file
let from_file = jsonfixer::from_file("broken.json", options).unwrap();
```

## 配置选项 / Configuration Options

```rust
use jsonfixer::JsonRepairOptions;

let options = JsonRepairOptions {
    return_objects: true,      // 返回解析后的对象而非字符串 / Return parsed objects instead of strings
    skip_json_loads: false,    // 跳过标准 JSON 解析器的首次尝试 / Skip first attempt with standard JSON parser
    logging: true,            // 启用修复日志 / Enable repair logs
    ensure_ascii: false,      // 转义非 ASCII 字符 / Escape non-ASCII characters
    stream_stable: false,     // 保持流式 JSON 的修复结果稳定 / Keep streaming JSON repairs stable
    chunk_length: 1024,       // 文件处理的块长度 / Chunk length for file processing
    indent: Some(2),          // 输出格式的缩进 / Indentation for output formatting
    ..Default::default()
};
```

## 示例 / Example

修复前 / Before repair：
```json
{
    name: John,
    age: 30,
    hobbies: ["reading", "swimming",]
    city: New York
}
```

修复后 / After repair：
```json
{
  "name": "John",
  "age": 30,
  "hobbies": [
    "reading",
    "swimming"
  ],
  "city": "New York"
}
```

## 许可证 / License

本项目采用 MIT 许可证 - 详情请查看 [LICENSE](LICENSE) 文件。
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 致谢 / Acknowledgments

灵感来源于 Py 版本的 json_repair 项目。
Inspired by the Python version of json_repair project.
