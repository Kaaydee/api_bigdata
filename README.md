# Usage Instructions

## Installation

```
pip install -r requirement.txt
python main.py
```

## Access FastAPI UI

Open in browser:

```
ip:8127/docs
```

## POST Request Body Example

```json
{
  "data": [
    [820, 18, 150, 55, 610],
    [830, 17, 152, 56, 620],
    [840, 16, 155, 58, 630],
    [850, 15, 158, 60, 640],
    [860, 14, 160, 62, 650]
  ]
}
```

## cURL Example

```
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
}'
```
