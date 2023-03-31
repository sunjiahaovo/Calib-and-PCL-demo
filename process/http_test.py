from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib
import json

'''========【http端口服务】========'''
class HttpHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path,args=urllib.parse.splitquery(self.path)
        self._response(path, args)
    def do_POST(self):
        args = self.rfile.read(int(self.headers['content-length'])).decode("utf-8")
        self._response(self.path, args)
    def _response(self, path, args):
        # 组装参数为字典
        if args:
            args=urllib.parse.parse_qs(args).items()
            args=dict([(k,v[0]) for k,v in args])
        else:
            args={}
        # 返回结果
        result={"success":True}
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

'''========【执行入口】========'''
if __name__ == '__main__':
    #开启http服务，设置监听ip和端口
    httpd = HTTPServer(('192.168.1.105', 9999), HttpHandler)
    httpd.serve_forever()