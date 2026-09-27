# Security Policy

请不要在公开 issue 中提交 API key、原始私有题目、日志或包含本机路径的截图。

若发现凭据泄露，请立即在对应 provider 处撤销或轮换密钥。仅删除当前分支中的文件
不能清除 Git 历史、fork、缓存或已下载副本。随后应清理历史并重新运行
`scripts/audit_release.py`。

本地审核服务器不提供认证，默认仅监听 loopback。若要通过网络访问，调用方必须
自行提供身份验证、TLS、防火墙和反向代理保护。
