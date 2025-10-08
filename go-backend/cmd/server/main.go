package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime"
	"net/http"
	"path/filepath"
	"time"
)

type detectRequest struct {
	Image string `json:"image"`
}
type detectResponse struct {
	Recognized bool    `json:"recognized"`
	Name       string  `json:"name"`
	Confidence float64 `json:"confidence"`
	Width      int     `json:"width"`
	Height     int     `json:"height"`
	Timestamp  string  `json:"timestamp"`
	Error      string  `json:"error,omitempty"`
}

func main() {
	// 提供静态网页
	fs := http.FileServer(http.Dir("./internal/web"))
	http.Handle("/", fs)

	// 图片上传接口，转发到 Python 服务
	http.HandleFunc("/api/detect", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if err := r.ParseMultipartForm(32 << 20); err != nil {
			http.Error(w, "parse form: "+err.Error(), http.StatusBadRequest)
			return
		}
		file, header, err := r.FormFile("image")
		if err != nil {
			http.Error(w, "missing image: "+err.Error(), http.StatusBadRequest)
			return
		}
		defer file.Close()
		data, _ := io.ReadAll(file)
		mimeType := mime.TypeByExtension(filepath.Ext(header.Filename))
		if mimeType == "" {
			mimeType = "image/jpeg"
		}
		b64 := base64.StdEncoding.EncodeToString(data)
		dataURL := fmt.Sprintf("data:%s;base64,%s", mimeType, b64)
		body, _ := json.Marshal(detectRequest{Image: dataURL})

		client := &http.Client{Timeout: 5 * time.Second}
		resp, err := client.Post("http://127.0.0.1:5000/detect", "application/json", bytes.NewReader(body))
		if err != nil {
			http.Error(w, "python service error: "+err.Error(), http.StatusBadGateway)
			return
		}
		defer resp.Body.Close()
		w.Header().Set("Content-Type", "application/json")
		io.Copy(w, resp.Body)
	})

	log.Println("Go backend running at http://127.0.0.1:8080")
	http.ListenAndServe("127.0.0.1:8080", nil)
}
