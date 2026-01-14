// 数据模型
struct Message: Identifiable {
    let id = UUID()
    let text: String
    let isUser: Bool
    let timestamp = Date()
}

struct Agent: Identifiable {
    let id = UUID()
    let name: String
    let description: String
}

struct APIResponse: Decodable {
    let success: Bool
    let output: String?
    let error: String?
}
