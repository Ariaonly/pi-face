// go-backend/cmd/server/main.go
package main

import (
	"log"
	"net/http"
	"os"

	"go-backend/internal/api"
    "go-backend/internal/store"
)

func main() {
	// 数据库存放在 go-backend/data/face.db
	st, err := store.Open("data")
	if err != nil {
		log.Fatal(err)
	}
	// Python 识别服务基址
	pyBase := env("PY_BASE", "http://127.0.0.1:5000")

	srv, err := api.NewServer(st, pyBase)
	if err != nil {
		log.Fatal(err)
	}

	addr := env("ADDR", "127.0.0.1:8080")
	log.Printf("Go backend running at http://%s\n", addr)
	if err := http.ListenAndServe(addr, srv.Mux); err != nil {
		log.Fatal(err)
	}
}

func env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
