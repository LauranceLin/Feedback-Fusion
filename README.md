# Feedback-Fusion

## 專案簡介
這個系統可以分析 Google 評論並通過客戶撰寫的內容讓商家了解自己的優勢和劣勢。

## 執行流程
1. 從 [Apify](https://apify.com/compass/google-maps-reviews-scraper) 下載所需的評論。
2. 將下載的評論整理成只有 'star' 和 'text' 的欄位。(optional)
3. 修改程式碼裡面的 stopwords_path 和 file_path。
4. 在 command line 執行該檔案。
5. 程式會需要使用者輸入要了解該商將的優勢(p)還是劣勢(n)。
6. 等待顯示結果。

## 結果演示
以公館新馬辣火鍋店的 Google 評論為例。\
自定義字典範例：\
<img src="https://github.com/user-attachments/assets/2d4dc5c9-2679-4ea2-8575-577bf4b7f620" alt="drawing" width="600"/>

分群結果範例：\
<img src="https://github.com/user-attachments/assets/a9388591-7a71-42a7-9adb-68e9b6cc45e7" alt="drawing" width="500"/>
