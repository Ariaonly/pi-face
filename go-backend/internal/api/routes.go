// go-backend/internal/api/routes.go
package api

import "net/http"

func (s *Server) routes() {
	// 页面
	s.Mux.HandleFunc("/", s.handleHome)

	// 反向代理到 Python
	s.Mux.HandleFunc("/video_feed", s.handleVideo)
	s.Mux.HandleFunc("/api/py/last", s.handlePyLast)

	// 记录 CRUD（目前只做必要的两条）
	s.Mux.HandleFunc("/api/records", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			s.handleInsertRecord(w, r)
		case http.MethodGet:
			s.handleListRecords(w, r)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	})
}
