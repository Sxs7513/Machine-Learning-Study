@echo off
@title "git定时提交"
echo =======================================================
echo          Starting automatic git commit push
echo =======================================================

D:
cd D:\Users\xssong\study\machine-learing\python
git add .
git commit -m "add"
git push origin master

echo done
pause
