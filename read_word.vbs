Option Explicit

On Error Resume Next

Dim WordApp, WordDoc, FileSystem, OutputFile
Dim TextContent

Set FileSystem = CreateObject("Scripting.FileSystemObject")
Set OutputFile = FileSystem.CreateTextFile("芦若禾简历内容.txt", True, True)

' 创建Word应用程序对象
Set WordApp = CreateObject("Word.Application")
If Err.Number <> 0 Then
    OutputFile.WriteLine "错误: 无法创建Word应用程序对象"
    OutputFile.WriteLine "请确保已安装Microsoft Word"
    OutputFile.Close
    WScript.Quit(1)
End If

WordApp.Visible = False

' 打开文档
Set WordDoc = WordApp.Documents.Open("C:\Users\霍冠华\Desktop\简历\芦若禾简历.docx")
If Err.Number <> 0 Then
    OutputFile.WriteLine "错误: 无法打开文档文件"
    OutputFile.WriteLine "错误代码: " & Err.Number
    OutputFile.WriteLine "错误描述: " & Err.Description
    WordApp.Quit
    OutputFile.Close
    WScript.Quit(1)
End If

' 获取文档内容
TextContent = WordDoc.Content.Text

' 写入文件
OutputFile.WriteLine "=== 芦若禾简历内容 ==="
OutputFile.WriteLine ""
OutputFile.WriteLine TextContent

' 清理
WordDoc.Close
WordApp.Quit
Set WordDoc = Nothing
Set WordApp = Nothing

OutputFile.WriteLine ""
OutputFile.WriteLine "=== 读取完成 ==="
OutputFile.Close

WScript.Echo "文档内容已保存到: 芦若禾简历内容.txt"