# AIEN16Team2FaceMesh
FaceMesh

github

終端機進行設置：
$ git config --global user.name name
$ git config --global user.email Email
#配置你的 username 與 email　(與 GitHub 帳號、Email 一致)

#########################
這裡我簡化本篇流程：

開新資料夾，git init 建立數據庫
新增一個 index.html 檔案
加入索引：git add .
提交版本：git commit -m "新增網頁"
觀看歷史紀錄：git log，並會看到一個版本紀錄



#############################################

一、前置作業
1.安裝 Git

2. $ git clone https://github.com/we1tingyu/AIEN16Team2FaceMesh.git

3. $ git remote add origin https://github.com/we1tingyu/AIEN16Team2FaceMesh.git
#加入遠端節點
[補充說明] origin
origin 只是一個遠端的代名詞，會取這個名字是因為如果從 Server Clone 下來會是這個代稱。所以如果把上面那句指令翻成中文的話：「為目前所在本地端檔案庫增加一個叫做 origin 的遠端檔案庫」

4. $ git branch dev
#切換到dev分支

5. $ git branch
#查看目前分支

6. $ git add 檔案名稱，上傳檔案到暫存庫

7. $ git status 查看狀態
#您會發現該檔案狀態本來是 Untracked files，變成是Changes to be committed，這樣就表示有成功加入到索引。
#Changes to be committed的意思是，放在索引的檔案即將會被提交成一個新版本(commit)。

8.$ git commit -m "<填寫版本資訊>"
#您可以再次使用 git status 觀察，會獲得以下資訊：
#1 file changed, 0 insertions(+), 0 deletions(-)

9. $ git log 
#查看新增版本

10. $ git push -u origin master
將本地端檔案庫（local repo）推上一個叫做 origin 的遠端檔案庫（remote repo）master 分支

https://w3c.hexschool.com/img/%E8%9E%A2%E5%B9%95%E6%88%AA%E5%9C%96_2019-11-16_22.03.067qvx7.png




