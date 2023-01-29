# **SportTotal test task**

Command to run the app:
```
docker build -t server_trt .
docker run --runtime nvidia -p 3000:3000 -it server_trt
```

To get the prediction run this command:
```
curl -X POST -H "Content-Type: image/jpeg" --data-binary "@/path/to/image" http://jetson_ip:port
```