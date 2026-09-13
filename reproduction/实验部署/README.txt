论文 LaTeX 源码编译包

主文件：main.tex；编译引擎：pdfLaTeX。

Overleaf：上传 ZIP 为新项目，主文件选 main.tex，编译器选 pdfLaTeX，然后 Recompile。
本地：安装依赖齐全的 TeX Live / MiKTeX，在本目录依次运行两次：
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
也可运行 latexmk -pdf main.tex。
Windows 可双击 compile.bat（需要 pdflatex 已安装并加入 PATH）。

已包含正文实际引用的所有表格、图片、参考文献及 elsarticle.cls。
参考文献由 bibliography.tex 直接载入，无需 BibTeX。
标准宏包和字体由 TeX 发行版提供，包括 newtxtext/newtxmath、TikZ、hyperref 等。
精简 MiKTeX 用户请允许自动安装缺失宏包，首次安装可能需要联网。
无需 Python、实验数据或模型文件。请保持子目录与文件名不变。
main.pdf 为本包源码编译生成的预览。
elsarticle.cls 为原样随附的第三方文档类，版权和许可见文件头部。
