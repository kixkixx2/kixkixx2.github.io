/**
 * Shared Chatbot Component
 * Provides drag, expand/minimize, and chat functionality
 */

// Chatbot state
let isDragging = false;
let currentX;
let currentY;
let initialX;
let initialY;
let xOffset = 0;
let yOffset = 0;
let isMinimized = false;

// Elements - will be initialized after DOM loads
let chatButton;
let chatWindow;
let chatHeader;
let closeChatBtn;
let minimizeChatBtn;
let chatInput;
let sendChatBtn;
let chatMessages;

// Restore chat state on page load
function restoreChatState() {
    const chatState = sessionStorage.getItem('chatState');
    const chatHistory = sessionStorage.getItem('chatHistory');
    
    // Restore chat history first
    if (chatHistory) {
        try {
            const messages = JSON.parse(chatHistory);
            if (messages && messages.length > 0) {
                chatMessages.innerHTML = ''; // Clear default message only if we have history
                messages.forEach(msg => {
                    addChatMessage(msg.text, msg.sender, false); // false = don't save to storage
                });
            }
        } catch (error) {
            console.error('Error restoring chat history:', error);
        }
    }
    
    // Then restore window state
    if (chatState) {
        try {
            const state = JSON.parse(chatState);
            
            // Restore window visibility
            if (state.isOpen) {
                chatWindow.classList.add('active');
            }
            
            // Restore minimized state
            if (state.isMinimized) {
                isMinimized = true;
                toggleMinimize();
            }
            
            // Restore position
            if (state.xOffset !== undefined && state.yOffset !== undefined) {
                xOffset = state.xOffset;
                yOffset = state.yOffset;
                setTranslate(xOffset, yOffset, chatWindow);
            }
        } catch (error) {
            console.error('Error restoring chat state:', error);
        }
    }
}

// Save chat state
function saveChatState() {
    const state = {
        isOpen: chatWindow.classList.contains('active'),
        isMinimized: isMinimized,
        xOffset: xOffset,
        yOffset: yOffset
    };
    sessionStorage.setItem('chatState', JSON.stringify(state));
}

// Save chat history
function saveChatHistory() {
    const messages = Array.from(chatMessages.querySelectorAll('.chat-message')).map(msg => {
        const isUser = msg.classList.contains('user-message');
        const textContent = msg.querySelector('.message-content').textContent;
        return {
            text: textContent,
            sender: isUser ? 'user' : 'ai'
        };
    });
    sessionStorage.setItem('chatHistory', JSON.stringify(messages));
}

// Minimize toggle function
function toggleMinimize() {
    isMinimized = !isMinimized;
    if (isMinimized) {
        chatWindow.classList.add('minimized');
        minimizeChatBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18"></rect></svg>';
        minimizeChatBtn.title = 'Restore';
    } else {
        chatWindow.classList.remove('minimized');
        minimizeChatBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="5" y1="12" x2="19" y2="12"></line></svg>';
        minimizeChatBtn.title = 'Minimize';
    }
    saveChatState();
}

// Drag start function
function dragStart(e) {
    if (e.target.closest('button')) {
        // Don't drag if clicking on buttons
        return;
    }
    
    if (e.type === 'touchstart') {
        initialX = e.touches[0].clientX - xOffset;
        initialY = e.touches[0].clientY - yOffset;
    } else {
        initialX = e.clientX - xOffset;
        initialY = e.clientY - yOffset;
    }

    if (e.target === chatHeader || e.target.closest('.chat-header')) {
        isDragging = true;
        chatWindow.classList.add('dragging');
    }
}

function drag(e) {
    if (isDragging) {
        e.preventDefault();
        
        if (e.type === 'touchmove') {
            currentX = e.touches[0].clientX - initialX;
            currentY = e.touches[0].clientY - initialY;
        } else {
            currentX = e.clientX - initialX;
            currentY = e.clientY - initialY;
        }

        xOffset = currentX;
        yOffset = currentY;

        setTranslate(currentX, currentY, chatWindow);
    }
}

function dragEnd(e) {
    if (isDragging) {
        initialX = currentX;
        initialY = currentY;
        isDragging = false;
        chatWindow.classList.remove('dragging');
        saveChatState(); // Save position when drag ends
    }
}

function setTranslate(xPos, yPos, el) {
    el.style.transform = `translate(${xPos}px, ${yPos}px)`;
}

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    // Add user message to chat
    addChatMessage(message, 'user');
    chatInput.value = '';

    // Show typing indicator
    showTypingIndicator();

    try {
        // Send to API
        const response = await fetch(`${API_BASE_URL}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                cluster_id: null
            })
        });

        const data = await response.json();

        // Remove typing indicator
        removeTypingIndicator();

        if (data.success) {
            addChatMessage(data.response, 'ai');
        } else {
            addChatMessage(`Error: ${data.error}`, 'ai');
        }
    } catch (error) {
        removeTypingIndicator();
        addChatMessage('Sorry, I encountered an error connecting to the AI service. Please make sure the API server is running.', 'ai');
        console.error('Chat error:', error);
    }
}

function addChatMessage(text, sender, shouldSave = true) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender}-message`;
    
    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    
    if (sender === 'ai') {
        avatarDiv.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"></circle>
                <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path>
            </svg>
        `;
    } else {
        avatarDiv.textContent = 'U';
    }
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = text;
    
    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(contentDiv);
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Save chat history if needed
    if (shouldSave) {
        saveChatHistory();
    }
}

function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.id = 'typingIndicator';
    typingDiv.innerHTML = `
        <span>AI is thinking</span>
        <div class="typing-dots">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Initialize all elements
function initializeChatElements() {
    chatButton = document.getElementById('chatButton');
    chatWindow = document.getElementById('chatWindow');
    chatHeader = document.getElementById('chatHeader');
    closeChatBtn = document.getElementById('closeChatBtn');
    minimizeChatBtn = document.getElementById('minimizeChatBtn');
    chatInput = document.getElementById('chatInput');
    sendChatBtn = document.getElementById('sendChatBtn');
    chatMessages = document.getElementById('chatMessages');
    
    // Verify all elements exist
    if (!chatButton || !chatWindow || !chatHeader || !closeChatBtn || 
        !minimizeChatBtn || !chatInput || !sendChatBtn || !chatMessages) {
        console.error('❌ Chatbot Error: Some elements are missing!');
        console.log('chatButton:', chatButton);
        console.log('chatWindow:', chatWindow);
        console.log('chatHeader:', chatHeader);
        console.log('closeChatBtn:', closeChatBtn);
        console.log('minimizeChatBtn:', minimizeChatBtn);
        console.log('chatInput:', chatInput);
        console.log('sendChatBtn:', sendChatBtn);
        console.log('chatMessages:', chatMessages);
        return false;
    }
    
    console.log('✅ All chatbot elements found');
    return true;
}

// Initialize chatbot and restore state
document.addEventListener('DOMContentLoaded', () => {
    console.log('🔄 DOM loaded, initializing chatbot...');
    
    // Initialize elements first
    if (!initializeChatElements()) {
        console.error('❌ Failed to initialize chatbot - missing elements');
        return;
    }
    
    // Setup event listeners
    setupChatEventListeners();
    
    // Restore previous state
    setTimeout(() => {
        restoreChatState();
        console.log('✅ Chatbot initialized - Draggable, Expandable, and Persistent!');
    }, 100);
});

// Setup all event listeners
function setupChatEventListeners() {
    // Chat button toggle
    chatButton.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        
        console.log('💬 Chat button clicked!');
        
        // Toggle active class
        const wasActive = chatWindow.classList.contains('active');
        chatWindow.classList.toggle('active');
        const isActive = chatWindow.classList.contains('active');
        
        console.log('Was active:', wasActive, '→ Now active:', isActive);
        console.log('Element classes:', chatWindow.className);
        console.log('Computed opacity:', window.getComputedStyle(chatWindow).opacity);
        console.log('Computed visibility:', window.getComputedStyle(chatWindow).visibility);
        console.log('Computed transform:', window.getComputedStyle(chatWindow).transform);
        
        if (isActive && isMinimized) {
            toggleMinimize();
        }
        saveChatState();
    });

    // Close chat
    closeChatBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        
        chatWindow.classList.remove('active');
        console.log('❌ Chat closed via close button');
        saveChatState();
    });

    // Minimize chat (toggle hide/unhide)
    minimizeChatBtn.addEventListener('click', () => {
        toggleMinimize();
    });

    // Send message
    sendChatBtn.addEventListener('click', sendMessage);
    
    // Enter key to send
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Drag functionality - mouse
    chatHeader.addEventListener('mousedown', dragStart);
    document.addEventListener('mousemove', drag);
    document.addEventListener('mouseup', dragEnd);
    
    // Drag functionality - touch
    chatHeader.addEventListener('touchstart', dragStart);
    document.addEventListener('touchmove', drag);
    document.addEventListener('touchend', dragEnd);
    
    console.log('✅ Event listeners attached');
}


