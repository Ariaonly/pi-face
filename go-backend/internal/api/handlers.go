// go-backend/internal/api/handlers.go
package api

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strconv"
	"time"

	"go-backend/internal/store"

)

type Server struct {
	Store      *store.Store
	PyBase     *url.URL       // Python 服务基地址
	VideoProxy *httputil.ReverseProxy
	LastProxy  *httputil.ReverseProxy
	Mux        *http.ServeMux
}

func NewServer(st *store.Store, pyBase string) (*Server, error) {
	u, err := url.Parse(pyBase)
	if err != nil {
		return nil, err
	}
	video, err := url.Parse(pyBase + "/video_feed")
	if err != nil {
		return nil, err
	}
	last, err := url.Parse(pyBase + "/api/last")
	if err != nil {
		return nil, err
	}
	s := &Server{
		Store:      st,
		PyBase:     u,
		VideoProxy: httputil.NewSingleHostReverseProxy(video),
		LastProxy:  httputil.NewSingleHostReverseProxy(last),
		Mux:        http.NewServeMux(),
	}
	s.routes()
	return s, nil
}

// --- DTOs ---

type RecordDTO struct {
	Name       string  `json:"name"`
	Confidence float64 `json:"confidence"`
	Timestamp  string  `json:"timestamp"` // ISO8601
}

// --- Handlers ---

func (s *Server) handleHome(w http.ResponseWriter, r *http.Request) {
	http.ServeFile(w, r, "internal/web/index.html")
}

// 反向代理：把 /video_feed 转发到 Python
func (s *Server) handleVideo(w http.ResponseWriter, r *http.Request) {
	// MJPEG 流需要直通
	s.VideoProxy.ServeHTTP(w, r)
}

// 反向代理：把 /api/py/last 转发到 Python
func (s *Server) handlePyLast(w http.ResponseWriter, r *http.Request) {
	s.LastProxy.ServeHTTP(w, r)
}

// POST /api/records   ← Python 识别成功后回调
func (s *Server) handleInsertRecord(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()
	var dto RecordDTO
	if err := json.NewDecoder(r.Body).Decode(&dto); err != nil {
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}
	if dto.Timestamp == "" {
		dto.Timestamp = time.Now().UTC().Format(time.RFC3339)
	}
	err := s.Store.InsertRecord(r.Context(), store.Record{
		Name:       dto.Name,
		Confidence: dto.Confidence,
		Timestamp:  dto.Timestamp,
	})
	if err != nil {
		log.Println("insert error:", err)
		http.Error(w, "db error", http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusCreated)
	io.WriteString(w, `{"ok":true}`)
}

// GET /api/records?limit=50  ← Web 前端查询访客记录
func (s *Server) handleListRecords(w http.ResponseWriter, r *http.Request) {
	limit := 50
	if q := r.URL.Query().Get("limit"); q != "" {
		if n, err := strconv.Atoi(q); err == nil && n > 0 && n <= 500 {
			limit = n
		}
	}
	recs, err := s.Store.ListRecords(context.Background(), limit)
	if err != nil {
		log.Println("list error:", err)
		http.Error(w, "db error", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(recs)
}
