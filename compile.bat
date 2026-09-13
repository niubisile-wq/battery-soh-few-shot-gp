@echo off
setlocal
cd /d "%~dp0"
where pdflatex >nul 2>nul
if errorlevel 1 goto missing
set "PAPER_TEX_FLAGS=-interaction=nonstopmode -halt-on-error"
pdflatex --version | findstr /I "MiKTeX" >nul
if not errorlevel 1 set "PAPER_TEX_FLAGS=--disable-installer %PAPER_TEX_FLAGS%"
for %%D in (main supplementary) do (
  for /L %%I in (1,1,3) do (
    pdflatex %PAPER_TEX_FLAGS% %%D.tex
    if errorlevel 1 goto failed
  )
)
echo Build complete: main.pdf and supplementary.pdf
pause
exit /b 0
:missing
echo Install TeX Live or MiKTeX and add pdflatex to PATH. See README.txt.
pause
exit /b 1
:failed
echo Build failed. Read the log and install missing packages, including newtx if requested.
echo See README.txt for dependencies.
pause
exit /b 1
