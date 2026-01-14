import Foundation

class ViewModel: ObservableObject {
    @Published var messages: [Message] = []
    @Published var agents: [Agent] = []
    @Published var selectedAgent: Agent?
    @Published var isLoading = false
    
    init() {
        // 初始化时添加欢迎消息
        let welcomeMessage = Message(text: "您好！我是iFlow AI助手，很高兴为您服务。您可以向我发送命令，或者点击右下角的智能体按钮选择特定功能的智能体。", isUser: false)
        messages.append(welcomeMessage)
    }
    
    // 加载智能体列表
    func loadAgents() async {
        DispatchQueue.main.async {
            self.isLoading = true
        }
        
        do {
            let loadedAgents = try await APIService.shared.getAgents()
            DispatchQueue.main.async {
                self.agents = loadedAgents
                self.isLoading = false
            }
        } catch {
            DispatchQueue.main.async {
                self.isLoading = false
                let errorMessage = Message(text: "无法加载智能体列表: \(error.localizedDescription)", isUser: false)
                self.messages.append(errorMessage)
            }
        }
    }
    
    // 选择智能体
    func selectAgent(_ agent: Agent) {
        selectedAgent = agent
        let message = Message(text: "已选择智能体: \(agent.name)\n\(agent.description)\n\n您可以开始与该智能体交互了。", isUser: false)
        messages.append(message)
    }
    
    // 使用智能体执行命令
    func executeAgentCommand(prompt: String) async {
        guard let agent = selectedAgent else {
            let errorMessage = Message(text: "请先选择一个智能体", isUser: false)
            messages.append(errorMessage)
            return
        }
        
        let userMessage = Message(text: prompt, isUser: true)
        messages.append(userMessage)
        
        do {
            let response = try await APIService.shared.executeAgent(name: agent.name, prompt: prompt)
            let aiMessage = Message(text: response, isUser: false)
            messages.append(aiMessage)
        } catch {
            let errorMessage = Message(text: "执行智能体命令失败: \(error.localizedDescription)", isUser: false)
            messages.append(errorMessage)
        }
    }
    
    // 直接执行命令（不使用特定智能体）
    func executeCommand(prompt: String) async {
        let userMessage = Message(text: prompt, isUser: true)
        messages.append(userMessage)
        
        do {
            let response = try await APIService.shared.executeCommand(prompt: prompt)
            let aiMessage = Message(text: response, isUser: false)
            messages.append(aiMessage)
        } catch {
            let errorMessage = Message(text: "执行命令失败: \(error.localizedDescription)", isUser: false)
            messages.append(errorMessage)
        }
    }
}
