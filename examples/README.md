# 自包含示例

`tiny/` 是一个完整、极小的 BookShelf 设计：它包含 `.aux`、`.nodes`、`.nets`、`.pl`、`.scl`
和 `.wts` 六类输入文件。它只用于快速体验和验证命令行流程，不依赖项目目录之外的任何文件。

运行方式：

```bash
./build/myplace examples/tiny/tiny.aux --output out/scratch/tiny-example --initial all
```

实际课程或个人设计只需将自己的 BookShelf `.aux` 文件作为第一个命令行参数传入即可。
