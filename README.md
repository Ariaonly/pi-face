# pi-face
```
├── go-backend/                  ← Go 服务端目录
│   ├── cmd/
│   │   └── server/
│   │       └── main.go          ← 启动入口
│   ├── internal/
│   │   ├── api/
│   │   │   ├── handlers.go      ← HTTP API 接口
│   │   │   └── routes.go        ← 路由注册
│   │   ├── service/
│   │   │   └── recognizer.go    ← 调用 Python 识别服务
│   │   ├── store/
│   │   │   └── db.go            ← SQLite 数据库操作
│   │   └── web/
│   │       └── index.html       ← 前端网页 (HTML+CSS)
│   ├── go.mod
│   ├── go.sum
│   └── data/
│       └── face.db              ← 数据库存放目录
│
├── py-recognizer/               ← Python 人脸识别服务
│   ├── main.py                  ← Flask 服务主程序
│   ├── recognizer.py            ← 识别逻辑
│   ├── models/                  ← 模型文件 (ArcFace, Dlib 等)
│   ├── faces/                   ← 已注册人脸照片
│   ├── embeddings/              ← 提取后的人脸特征 (numpy 文件)
│   ├── utils/
│   │   ├── tts.py               ← 语音反馈模块
│   │   └── camera.py            ← 摄像头读取模块
│   ├── requirements.txt         ← Python 依赖清单
│   └── logs/
│       └── service.log          ← 运行日志
│
├── run.sh                       ← 一键启动脚本（Python + Go）
└── README.md                    ← 项目说明文档
```

