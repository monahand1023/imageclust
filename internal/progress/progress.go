package progress

import "sync"

type Manager struct {
	clients map[chan string]bool
	mu      sync.RWMutex
}

func NewManager() *Manager {
	return &Manager{
		clients: make(map[chan string]bool),
	}
}

func (pm *Manager) Register(client chan string) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	pm.clients[client] = true
}

func (pm *Manager) Unregister(client chan string) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	delete(pm.clients, client)
	close(client)
}

func (pm *Manager) Broadcast(message string) {
	pm.mu.RLock()
	defer pm.mu.RUnlock()
	for client := range pm.clients {
		client <- message
	}
}

var Default = NewManager()
