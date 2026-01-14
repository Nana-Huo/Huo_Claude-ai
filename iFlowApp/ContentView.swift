import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = ViewModel()
    @State private var inputText = ""
    @State private var isAgentsSheetPresented = false
    
    var body: some View {
        NavigationView {
            VStack {
                // 聊天消息列表
                ScrollViewReader {proxy in
                    List(viewModel.messages) { message in
                        MessageView(message: message)
                    }
                    .listStyle(.plain)
                    .onChange(of: viewModel.messages.last?.id) { id in
                        if let id = id {
                            withAnimation {
                                proxy.scrollTo(id, anchor: .bottom)
                            }
                        }
                    }
                }
                
                // 输入区域
                HStack {
                    TextField("输入命令...", text: $inputText)
                        .textFieldStyle(.roundedBorder)
                        .padding(.leading)
                    
                    Button(action: {
                        sendMessage()
                    }) {
                        Text("发送")
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                    .padding(.trailing)
                    .disabled(inputText.isEmpty)
                    
                    Button(action: {
                        isAgentsSheetPresented.toggle()
                    }) {
                        Image(systemName: "person.2.fill")
                            .font(.title2)
                            .padding(.trailing)
                    }
                }
                .padding(.bottom)
            }
            .navigationTitle("iFlow AI")
            .sheet(isPresented: $isAgentsSheetPresented) {
                AgentsView(viewModel: viewModel, onDismiss: { isAgentsSheetPresented = false })
            }
            .task {
                await viewModel.loadAgents()
            }
        }
    }
    
    private func sendMessage() {
        guard !inputText.isEmpty else { return }
        
        let userMessage = Message(text: inputText, isUser: true)
        viewModel.messages.append(userMessage)
        
        let message = inputText
        inputText = ""
        
        Task {
            do {
                let response = try await APIService.shared.executeCommand(prompt: message)
                let aiMessage = Message(text: response, isUser: false)
                viewModel.messages.append(aiMessage)
            } catch {
                let errorMessage = Message(text: "错误: \(error.localizedDescription)", isUser: false)
                viewModel.messages.append(errorMessage)
            }
        }
    }
}

// 消息视图
struct MessageView: View {
    let message: Message
    
    var body: some View {
        HStack {
            if message.isUser {
                Spacer()
                Text(message.text)
                    .padding(12)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(16)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 4)
            } else {
                Text(message.text)
                    .padding(12)
                    .background(Color.gray.opacity(0.2))
                    .foregroundColor(.black)
                    .cornerRadius(16)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 4)
                Spacer()
            }
        }
    }
}

// 智能体选择视图
struct AgentsView: View {
    @ObservedObject var viewModel: ViewModel
    let onDismiss: () -> Void
    
    var body: some View {
        NavigationView {
            List(viewModel.agents) { agent in
                VStack(alignment: .leading) {
                    Text(agent.name)
                        .font(.headline)
                    Text(agent.description)
                        .font(.subheadline)
                        .foregroundColor(.gray)
                        .lineLimit(2)
                }
                .onTapGesture {
                    viewModel.selectAgent(agent)
                    onDismiss()
                }
            }
            .navigationTitle("选择智能体")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("取消") {
                        onDismiss()
                    }
                }
            }
        }
    }
}
