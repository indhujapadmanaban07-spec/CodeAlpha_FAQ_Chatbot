import streamlit as st
import pandas as pd
import numpy as np
import re
import random
import time
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

# ==================== ENHANCED FAQ DATABASE ====================
faqs = {
    # CodeAlpha Internship
    "What is CodeAlpha?": {
        "answer": "🚀 CodeAlpha is an innovative tech education platform offering virtual internships in AI, Web Development, Data Science, and Cybersecurity with hands-on projects.",
        "category": "General",
        "tags": ["introduction", "overview"]
    },
    "How to get internship certificate?": {
        "answer": "📜 To earn your certificate: 1) Complete 2-3 domain projects 2) Upload to GitHub 3) Post on LinkedIn 4) Submit via Google Form. Digital certificate issued within 2 weeks.",
        "category": "Certificate",
        "tags": ["certificate", "completion"]
    },
    "What tasks should I complete?": {
        "answer": "🎯 AI Domain Tasks: 1) Language Translation Tool 2) FAQ Chatbot 3) Music Generation AI 4) Object Detection System. Complete any 2-3 for certification.",
        "category": "Tasks",
        "tags": ["projects", "tasks"]
    },
    "How to submit completed tasks?": {
        "answer": "📤 Submission Process: 1) GitHub repository with full code 2) LinkedIn post tagging @CodeAlpha 3) Video demo 4) Google Form submission. All links required.",
        "category": "Submission",
        "tags": ["submission", "process"]
    },
    "Is the certificate free?": {
        "answer": "✅ Absolutely FREE! No hidden charges. Earn through project completion only. Certificate includes QR verification.",
        "category": "Certificate",
        "tags": ["free", "certificate"]
    },
    
    # Technical
    "What technologies are recommended?": {
        "answer": "💻 Python (TensorFlow/PyTorch), JavaScript (React/Node.js), or any stack you prefer. Focus on functionality over specific tech.",
        "category": "Technical",
        "tags": ["tech", "tools"]
    },
    
    # Timeline
    "What is the internship duration?": {
        "answer": "⏰ Flexible 4-8 weeks. Self-paced learning. Recommended: 4 weeks for optimal experience.",
        "category": "Timeline",
        "tags": ["duration", "time"]
    },
    
    # Support
    "Who can I contact for help?": {
        "answer": "📞 Support: Email → services@codealpha.tech | WhatsApp → +91 8062923611 | Website → codealpha.tech",
        "category": "Support",
        "tags": ["contact", "help"]
    }
}

# ==================== INTELLIGENT CHATBOT ====================
class SmartChatbot:
    def __init__(self, faq_data):
        self.faq_data = faq_data
        self.questions = list(faq_data.keys())
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        self._prepare_model()
        self.conversation_history = []
    
    def _simple_preprocess(self, text):
        """Clean text without NLTK"""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces
        text = re.sub(r'\s+', ' ', text)      # Remove extra spaces
        return text.strip()
    
    def _prepare_model(self):
        """Prepare TF-IDF model"""
        processed_questions = [self._simple_preprocess(q) for q in self.questions]
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_questions)
    
    def get_response(self, user_question):
        """Get smart response with alternatives"""
        # Clean user question
        processed_q = self._simple_preprocess(user_question)
        
        # Vectorize and find matches
        user_vector = self.vectorizer.transform([processed_q])
        similarities = cosine_similarity(user_vector, self.tfidf_matrix)[0]
        
        # Get top 3 matches
        top_indices = np.argsort(similarities)[-3:][::-1]
        best_idx = top_indices[0]
        best_score = similarities[best_idx]
        
        if best_score > 0.25:  # Good match
            matched_q = self.questions[best_idx]
            faq = self.faq_data[matched_q]
            
            # Prepare response
            response = {
                "answer": faq["answer"],
                "confidence": min(best_score * 1.5, 0.99),  # Boost confidence
                "category": faq["category"],
                "matched_question": matched_q,
                "alternatives": []
            }
            
            # Add alternatives
            for idx in top_indices[1:]:
                if similarities[idx] > 0.2:
                    response["alternatives"].append({
                        "question": self.questions[idx],
                        "similarity": similarities[idx]
                    })
            
            # Log conversation
            self.conversation_history.append({
                "timestamp": datetime.now(),
                "user": user_question,
                "matched": matched_q,
                "confidence": best_score,
                "category": faq["category"]
            })
            
            return response
        else:
            # Fallback with suggestions
            suggestions = random.sample([
                "What is CodeAlpha?",
                "How to get certificate?",
                "What tasks to complete?",
                "Is certificate free?"
            ], 3)
            
            return {
                "answer": "🤔 I'm not sure about that. Try asking about internship process, certificate details, or task requirements.",
                "confidence": 0,
                "category": "Unknown",
                "suggestions": suggestions
            }
    
    def get_stats(self):
        """Get conversation statistics"""
        if not self.conversation_history:
            return {"total": 0, "categories": {}}
        
        categories = {}
        for entry in self.conversation_history:
            cat = entry["category"]
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "total": len(self.conversation_history),
            "categories": categories,
            "avg_confidence": np.mean([e.get("confidence", 0) for e in self.conversation_history])
        }

# ==================== STREAMLIT UI ====================
def main():
    # Initialize chatbot
    chatbot = SmartChatbot(faqs)
    
    # Page config
    st.set_page_config(
        page_title="🤖 CodeAlpha AI Assistant",
        page_icon="💬",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stChatInput {border-radius: 20px;}
    .chat-message {padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;}
    .user-message {background: #e3f2fd; border-left: 4px solid #2196f3;}
    .bot-message {background: #f5f5f5; border-left: 4px solid #4caf50;}
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=100)
    with col2:
        st.title("🤖 CodeAlpha AI Assistant")
        st.markdown("### Your Smart Internship Guide")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📚 Knowledge Base", "📊 Insights"])
    
    with tab1:
        # Chat interface
        st.markdown("#### Ask me anything about CodeAlpha internships!")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm here to help with CodeAlpha internships. Ask me about certificates, tasks, submission, or anything else! 🚀"}
            ]
        
        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "confidence" in msg and msg["confidence"] > 0:
                    st.caption(f"Confidence: {msg['confidence']:.0%}")
        
        # Chat input
        if prompt := st.chat_input("Type your question..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Get bot response
            with st.spinner("Thinking..."):
                time.sleep(0.3)
                response = chatbot.get_response(prompt)
                
                # Format bot response
                bot_msg = f"**{response['answer']}**\n\n"
                
                if response["confidence"] > 0:
                    bot_msg += f"*Category: {response['category']}*\n"
                    
                    if response.get("alternatives"):
                        bot_msg += "\n**Related questions:**\n"
                        for alt in response["alternatives"][:2]:
                            bot_msg += f"• {alt['question']}\n"
                
                if response.get("suggestions"):
                    bot_msg += "\n**Try asking:**\n"
                    for sug in response["suggestions"]:
                        bot_msg += f"• {sug}\n"
                
                # Add bot message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": bot_msg,
                    "confidence": response["confidence"]
                })
            
            st.rerun()
    
    with tab2:
        st.header("📚 Complete FAQ Database")
        
        # Display by category
        categories = {}
        for q, data in faqs.items():
            cat = data["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((q, data["answer"], data["tags"]))
        
        for cat, items in categories.items():
            with st.expander(f"{cat} ({len(items)} FAQs)", expanded=True):
                for question, answer, tags in items:
                    st.markdown(f"""
                    **❓ {question}**
                    
                    {answer}
                    
                    *Tags: {', '.join(tags)}*
                    
                    ---
                    """)
    
    with tab3:
        st.header("📊 Chatbot Insights")
        
        stats = chatbot.get_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Queries", stats["total"])
        with col2:
            st.metric("Avg Confidence", f"{stats.get('avg_confidence', 0):.0%}")
        with col3:
            st.metric("Categories Used", len(stats["categories"]))
        
        # Category distribution
        if stats["categories"]:
            st.subheader("Category Distribution")
            fig = go.Figure(data=[go.Pie(
                labels=list(stats["categories"].keys()),
                values=list(stats["categories"].values()),
                hole=0.3
            )])
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent queries
        if chatbot.conversation_history:
            st.subheader("Recent Questions")
            recent = chatbot.conversation_history[-5:]
            for entry in reversed(recent):
                st.text(f"📝 {entry['user'][:50]}...")
                st.caption(f"Matched: {entry['matched']} | Confidence: {entry['confidence']:.0%}")
                st.divider()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
    <p>🤖 Powered by AI | 🎯 CodeAlpha Task 2 | 🚀 Smart FAQ Chatbot</p>
    <p>Need direct help? Contact: <strong>services@codealpha.tech</strong></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()