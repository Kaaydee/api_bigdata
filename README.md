# pip install -r requirement.txt
# python main.py

truy cập qua API ui FastAPI: 
      ip:8127/docs
Post: 
body json
{
  "data": [
    [820, 18, 150, 55, 610],
    [830, 17, 152, 56, 620],
    [840, 16, 155, 58, 630],
    [850, 15, 158, 60, 640],
    [860, 14, 160, 62, 650]
  ]
}
# api
curl -X 'POST' \
  'http://192.168.1.148:8127/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "data": [
    [820, 18, 150, 55, 610],
    [830, 17, 152, 56, 620],
    [840, 16, 155, 58, 630],
    [850, 15, 158, 60, 640],
    [860, 14, 160, 62, 650]
  ]
}
'