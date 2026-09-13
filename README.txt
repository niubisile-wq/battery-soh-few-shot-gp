论文工作稿（排版修订）— 2026-09-12

阅读文件
  main.pdf          正文（40页，含 Highlights）
  supplementary.pdf 补充材料（11页）

编译入口
  main.tex          正文，pdfLaTeX 编译
  supplementary.tex 补充材料，pdfLaTeX 编译
  两个文档独立编译，无需 BibTeX，无需训练代码或实验数据。

Overleaf
  上传源码 ZIP 为新项目。主文件选择 main.tex，编译器选择 pdfLaTeX。
  阅读/编译补充材料时，将主文件切换为 supplementary.tex。

本地 TeX Live / MiKTeX
  请先安装具备所需宏包与字体的 TeX 发行版，并将命令加入 PATH。
  在 main.tex 所在目录运行：
    latexmk -pdf main.tex
    latexmk -pdf supplementary.tex
  或分别对每个文件运行三次：
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    pdflatex -interaction=nonstopmode -halt-on-error supplementary.tex
  Windows 可双击 compile.bat；它会顺序编译两个文档。

依赖说明
  随包附有原样 elsarticle.cls，许可证见其文件头。
  其余标准宏包和字体由 TeX 发行版提供，主要包括：
  newtx（newtxtext.sty、newtxmath.sty 及所需字体）、amsmath、amssymb、
  booktabs、graphicx、array、float、xcolor、microtype、pgf/TikZ、placeins、hyperref、
  geometry，以及 elsarticle/newtx 的传递依赖。
  如提示 newtxtext.sty 或 newtxmath.sty 缺失，请通过发行版的包管理器安装
  newtx 及其依赖，然后重新编译。不要以包内已有 PDF 代替本机编译验证。
  compile.bat 在 MiKTeX 下关闭交互式自动安装；缺包时会停止，便于明确处理。
  本轮在本地 MiKTeX 25.12 / pdfLaTeX 下实际编译并检查。

证据与检查
  evidence/ 包含本轮新增表格对应的未舍入 CSV/JSON，并包含绘图用的保存查询数组，不包含原始测量数据集和模型。
  可选：python evidence/verify_public_evidence.py（只需 Python 标准库）。
  该脚本检查源摘要哈希、模型身份、数值派生和正文一致性；不重训、不做模型推理，
  不重新运行 bootstrap。历史审查记录与本轮检查的范围见 supplementary.pdf。

英文文件名和相对路径均需保持不变。旧稿和旧 ZIP 保留在原目录，本包是独立修订版本。

新增复现入口：evidence/REPRODUCTION.md。四张补充图可用 evidence/regenerate_figures.py 重新生成。




