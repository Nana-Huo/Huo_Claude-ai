import Foundation

class APIService {
    static let shared = APIService()
    private let baseURL = "http://localhost:4000"
    private let session = URLSession.shared
    
    private init() {}
    
    // 通用的GET请求
    private func getRequest<T: Decodable>(endpoint: String) async throws -> T {
        guard let url = URL(string: baseURL + endpoint) else {
            throw URLError(.badURL)
        }
        
        let (data, response) = try await session.data(from: url)
        
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        
        return try JSONDecoder().decode(T.self, from: data)
    }
    
    // 通用的POST请求
    private func postRequest<T: Decodable>(endpoint: String, body: Data) async throws -> T {
        guard let url = URL(string: baseURL + endpoint) else {
            throw URLError(.badURL)
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = body
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        
        return try JSONDecoder().decode(T.self, from: data)
    }
    
    // 健康检查
    func healthCheck() async throws -> Bool {
        let response: [String: Any] = try await getRequest(endpoint: "/health")
        return response["status"] as? String == "ok"
    }
    
    // 获取智能体列表
    func getAgents() async throws -> [Agent] {
        let response: APIResponse = try await getRequest(endpoint: "/agents")
        
        guard let output = response.output, response.success else {
            throw NSError(domain: "APIService", code: 0, userInfo: [NSLocalizedDescriptionKey: response.error ?? "获取智能体列表失败"])
        }
        
        // 解析输出字符串获取智能体信息
        var agents: [Agent] = []
        let lines = output.components(separatedBy: "\n")
        
        var currentAgent: String?
        var currentDescription: String?
        
        for line in lines {
            if line.contains("•") && line.contains("(") {
                // 处理智能体名称
                if let agent = currentAgent, let description = currentDescription {
                    agents.append(Agent(name: agent, description: description))
                }
                
                // 提取新智能体名称
                let components = line.components(separatedBy: ")")
                if components.count > 0 {
                    let namePart = components[0].replacingOccurrences(of: "• ", with: "").trimmingCharacters(in: .whitespaces)
                    let name = namePart.replacingOccurrences(of: "{1B}[32m", with: "").replacingOccurrences(of: "{1B}[0m", with: "")
                    currentAgent = name
                    currentDescription = ""
                }
            } else if line.contains("描述:") && currentAgent != nil {
                // 处理智能体描述
                let description = line.replacingOccurrences(of: "  描述:", with: "").trimmingCharacters(in: .whitespaces)
                currentDescription = description
            }
        }
        
        // 添加最后一个智能体
        if let agent = currentAgent, let description = currentDescription {
            agents.append(Agent(name: agent, description: description))
        }
        
        return agents
    }
    
    // 执行iFlow命令
    func executeCommand(prompt: String) async throws -> String {
        let body = try JSONEncoder().encode(["prompt": prompt])
        let response: APIResponse = try await postRequest(endpoint: "/execute", body: body)
        
        guard let output = response.output, response.success else {
            throw NSError(domain: "APIService", code: 0, userInfo: [NSLocalizedDescriptionKey: response.error ?? "执行命令失败"])
        }
        
        return output
    }
    
    // 执行特定智能体
    func executeAgent(name: String, prompt: String) async throws -> String {
        let body = try JSONEncoder().encode(["prompt": prompt])
        let response: APIResponse = try await postRequest(endpoint: "/agent/execute/\(name)", body: body)
        
        guard let output = response.output, response.success else {
            throw NSError(domain: "APIService", code: 0, userInfo: [NSLocalizedDescriptionKey: response.error ?? "执行智能体失败"])
        }
        
        return output
    }
}
