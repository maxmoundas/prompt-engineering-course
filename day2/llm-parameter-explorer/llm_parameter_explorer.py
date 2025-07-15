import streamlit as st
import openai
import anthropic
import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
import time

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="LLM Parameter Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_responses" not in st.session_state:
    st.session_state.api_responses = []


def load_model_costs() -> pd.DataFrame:
    """Load model costs from CSV file"""
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "model_costs.csv")
        costs_df = pd.read_csv(csv_path)
        return costs_df
    except Exception as e:
        st.error(f"Error loading model costs: {e}")
        return pd.DataFrame()


def calculate_cost(
    provider: str, model: str, input_tokens: int, output_tokens: int
) -> Dict[str, Any]:
    """Calculate the cost of an API call based on token usage"""
    costs_df = load_model_costs()

    if costs_df.empty:
        return {"success": False, "error": "Could not load model costs"}

    # Find the model in the costs dataframe
    model_costs = costs_df[
        (costs_df["provider"] == provider) & (costs_df["model"] == model)
    ]

    if model_costs.empty:
        return {
            "success": False,
            "error": f"No cost data found for {provider} model: {model}",
        }

    # Get cost per 1K tokens
    input_cost_per_1k = model_costs.iloc[0]["input_cost_per_1k_tokens"]
    output_cost_per_1k = model_costs.iloc[0]["output_cost_per_1k_tokens"]

    # Calculate costs
    input_cost = (input_tokens / 1000) * input_cost_per_1k
    output_cost = (output_tokens / 1000) * output_cost_per_1k
    total_cost = input_cost + output_cost

    return {
        "success": True,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "input_cost_per_1k": input_cost_per_1k,
        "output_cost_per_1k": output_cost_per_1k,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def get_openai_models():
    """Get list of available OpenAI models"""
    return [
        "gpt-4.1",
        "gpt-4.1-nano",
        "gpt-4o-mini",
    ]


def get_anthropic_models():
    """Get list of available Anthropic models"""
    return [
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-latest",
        "claude-3-5-haiku-latest",
    ]


def get_parameter_templates():
    """Get predefined parameter templates for different use cases"""
    return {
        "default": {
            "name": "Default",
            "description": "Balanced settings for general conversation and tasks. Good starting point for most use cases.",
            "parameters": {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 40,
                "max_tokens": 1000,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
            },
        },
        "brainstorming": {
            "name": "Brainstorming",
            "description": "High creativity settings for generating diverse ideas, creative writing, and exploring multiple possibilities. Encourages novel and unexpected outputs.",
            "parameters": {
                "temperature": 1.8,
                "top_p": 0.9,
                "top_k": 80,
                "max_tokens": 1500,
                "presence_penalty": 0.3,
                "frequency_penalty": 0.2,
            },
            "anthropic_temperature": 0.9,
        },
        "focused": {
            "name": "Focused",
            "description": "Low randomness settings for precise, factual responses, technical explanations, and when you need consistent, reliable outputs.",
            "parameters": {
                "temperature": 1.0,
                "top_p": 0.8,
                "top_k": 20,
                "max_tokens": 800,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
            },
        },
        "creative_writing": {
            "name": "Creative Writing",
            "description": "Optimized for storytelling, poetry, and artistic content. Balances creativity with coherence for engaging narratives.",
            "parameters": {
                "temperature": 1.4,
                "top_p": 0.85,
                "top_k": 60,
                "max_tokens": 2000,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.1,
            },
            "anthropic_temperature": 0.8,
        },
        "analysis": {
            "name": "Analysis",
            "description": "Settings for analytical tasks, data interpretation, and structured thinking. Produces well-reasoned, logical responses.",
            "parameters": {
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 30,
                "max_tokens": 1200,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
            },
        },
        "conversational": {
            "name": "Conversational",
            "description": "Natural conversation settings that mimic human-like dialogue. Good for chatbots and interactive applications.",
            "parameters": {
                "temperature": 0.8,
                "top_p": 0.95,
                "top_k": 50,
                "max_tokens": 600,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.05,
            },
        },
        "code_generation": {
            "name": "Code Generation",
            "description": "Optimized for programming tasks, code explanations, and technical documentation. Emphasizes accuracy and structure.",
            "parameters": {
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 15,
                "max_tokens": 1500,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
            },
        },
        "summarization": {
            "name": "Summarization",
            "description": "Settings for creating concise summaries, extracting key points, and condensing information effectively.",
            "parameters": {
                "temperature": 0.4,
                "top_p": 0.9,
                "top_k": 25,
                "max_tokens": 500,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
            },
        },
        "roleplay": {
            "name": "Roleplay",
            "description": "High creativity settings for character-based interactions, creative scenarios, and immersive storytelling experiences.",
            "parameters": {
                "temperature": 1.6,
                "top_p": 0.8,
                "top_k": 70,
                "max_tokens": 1800,
                "presence_penalty": 0.2,
                "frequency_penalty": 0.15,
            },
            "anthropic_temperature": 0.9,
        },
        "research": {
            "name": "Research",
            "description": "Balanced settings for research tasks, exploring topics, and generating comprehensive responses with multiple perspectives.",
            "parameters": {
                "temperature": 1.0,
                "top_p": 0.9,
                "top_k": 40,
                "max_tokens": 1600,
                "presence_penalty": 0.05,
                "frequency_penalty": 0.05,
            },
        },
    }


def apply_template(template_key: str, api_provider: str) -> Dict[str, Any]:
    """Apply a template to the session state"""
    templates = get_parameter_templates()
    if template_key in templates:
        template = templates[template_key]
        # Update session state with template parameters
        for param, value in template["parameters"].items():
            # Use appropriate temperature based on API provider
            if param == "temperature" and api_provider == "Anthropic":
                if "anthropic_temperature" in template:
                    st.session_state[f"param_{param}"] = template[
                        "anthropic_temperature"
                    ]
                else:
                    # Default to 0.7 for Anthropic if no specific value is set
                    st.session_state[f"param_{param}"] = 0.7
            else:
                st.session_state[f"param_{param}"] = value
        return template
    return None


def create_openai_payload(model: str, messages: List[Dict], **kwargs) -> Dict[str, Any]:
    """Create OpenAI API payload with all configurable parameters"""
    # Remove top_k as it's not supported by OpenAI
    if "top_k" in kwargs:
        del kwargs["top_k"]
    payload = {"model": model, "messages": messages, **kwargs}
    return payload


def create_anthropic_payload(
    model: str, messages: List[Dict], **kwargs
) -> Dict[str, Any]:
    """Create Anthropic API payload with all configurable parameters"""
    payload = {"model": model, "messages": messages, **kwargs}
    return payload


def call_openai_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Make OpenAI API call"""
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Extract parameters
        model = payload["model"]
        messages = payload["messages"]

        # Optional parameters
        optional_params = {}
        for key in [
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
            "response_format",
            "seed",
            "tools",
            "tool_choice",
            "user",
            "stream",
        ]:
            if key in payload and payload[key] is not None:
                optional_params[key] = payload[key]

        response = client.chat.completions.create(
            model=model, messages=messages, **optional_params
        )

        return {
            "success": True,
            "response": response,
            "usage": response.usage.model_dump() if response.usage else None,
            "model": response.model,
            "id": response.id,
            "created": response.created,
            "choices": [choice.model_dump() for choice in response.choices],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def call_anthropic_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Make Anthropic API call"""
    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # Extract parameters
        model = payload["model"]
        messages = payload["messages"]

        # Optional parameters
        optional_params = {}
        for key in [
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "system",
            "metadata",
            "stop_sequences",
            "stream",
        ]:
            if key in payload and payload[key] is not None:
                optional_params[key] = payload[key]

        response = client.messages.create(
            model=model, messages=messages, **optional_params
        )

        return {
            "success": True,
            "response": response,
            "usage": response.usage.model_dump() if response.usage else None,
            "model": response.model,
            "id": response.id,
            "type": response.type,
            "content": [content.model_dump() for content in response.content],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    st.title("🤖 LLM Parameter Explorer")
    st.markdown(
        "Explore and experiment with all configurable parameters for OpenAI and Anthropic LLM APIs"
    )

    # Sidebar for API selection and basic settings
    with st.sidebar:
        st.header("🔧 API Configuration")

        # API Provider Selection
        api_provider = st.selectbox(
            "Select API Provider",
            ["OpenAI", "Anthropic"],
            help="Choose between OpenAI and Anthropic APIs",
        )

        # Model Selection
        if api_provider == "OpenAI":
            models = get_openai_models()
        else:
            models = get_anthropic_models()

        selected_model = st.selectbox(
            "Select Model", models, help="Choose the LLM model to use"
        )

        # API Key Status
        if api_provider == "OpenAI":
            api_key = os.getenv("OPENAI_API_KEY")
        else:
            api_key = os.getenv("ANTHROPIC_API_KEY")

        if api_key:
            st.success("✅ API Key Found")
        else:
            st.error("❌ API Key Not Found")
            st.info(f"Please set {api_provider.upper()}_API_KEY in your .env file")

        st.divider()

        # Message Input
        st.header("💬 Message Input")
        user_message = st.text_area(
            "Your Message", placeholder="Enter your message here...", height=100
        )

        # System Message
        system_message = st.text_area(
            "System Message (Optional)",
            placeholder="Enter system message to set behavior...",
            height=80,
        )

        st.divider()

        # Generate Button
        if st.button("🚀 Generate Response", type="primary", use_container_width=True):
            if user_message.strip():
                st.session_state.generate_clicked = True
            else:
                st.error("Please enter a message")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("⚙️ Parameter Configuration")

        # Template Selector
        st.subheader("📋 Parameter Templates")
        templates = get_parameter_templates()

        # Create template options with descriptions
        template_options = {
            f"{templates[key]['name']} - {templates[key]['description']}": key
            for key in templates.keys()
        }

        selected_template = st.selectbox(
            "Choose a template to quickly set parameters:",
            options=list(template_options.keys()),
            index=0,
            help="Select a predefined template to automatically configure all parameters for a specific use case. Each template is optimized for different types of tasks and will set temperature, top_p, top_k, max_tokens, and penalty values accordingly.",
        )

        # Apply template button
        col_apply1, col_apply2 = st.columns([1, 1])
        with col_apply1:
            if st.button("Apply Template", type="secondary", use_container_width=True):
                template_key = template_options[selected_template]
                applied_template = apply_template(template_key, api_provider)
                if applied_template:
                    st.success(f"✅ Applied {applied_template['name']} template")
                    st.rerun()

        with col_apply2:
            if st.button(
                "Reset to Default", type="secondary", use_container_width=True
            ):
                # Reset all parameters to default values
                default_params = templates["default"]["parameters"]
                for param, value in default_params.items():
                    st.session_state[f"param_{param}"] = value
                st.success("✅ Reset to default parameters")
                st.rerun()

        st.divider()

        # Create tabs for different parameter categories
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "🎯 Core Parameters",
                "🎲 Sampling Parameters",
                "📏 Token Parameters",
                "🛠️ Advanced Parameters",
                "📊 Response Format",
            ]
        )

        with tab1:
            st.subheader("Core Parameters")

            # Temperature
            if "param_temperature" not in st.session_state:
                st.session_state.param_temperature = 1.0

            # Adjust temperature range based on API provider
            if api_provider == "Anthropic":
                max_temp = 1.0
                help_text = "Controls the randomness of the model's output. Lower values (0.0-0.5) make responses more focused and deterministic, while higher values (0.7-1.0) increase creativity and variety. A value of 0 makes the model always choose the most likely next token, while 1.0 allows for maximum diversity."
            else:  # OpenAI
                max_temp = 2.0
                help_text = "Controls the randomness of the model's output. Lower values (0.0-0.5) make responses more focused and deterministic, while higher values (1.0-2.0) increase creativity and variety. A value of 0 makes the model always choose the most likely next token, while 2.0 allows for much more diverse and unexpected outputs."

            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=max_temp,
                value=min(st.session_state.param_temperature, max_temp),
                step=0.1,
                key="temperature_slider",
                help=help_text,
            )
            st.session_state.param_temperature = temperature

            # Top P
            if "param_top_p" not in st.session_state:
                st.session_state.param_top_p = 1.0
            top_p = st.slider(
                "Top P (Nucleus Sampling)",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.param_top_p,
                step=0.01,
                key="top_p_slider",
                help="Controls diversity by considering only the most likely tokens whose cumulative probability exceeds this value. Lower values (0.1-0.5) make responses more focused by only considering high-probability tokens, while higher values (0.8-1.0) allow the model to consider a wider range of possibilities. Works together with temperature to control output randomness.",
            )
            st.session_state.param_top_p = top_p

            # Top K
            if "param_top_k" not in st.session_state:
                st.session_state.param_top_k = 40

            # Disable top_k for OpenAI as it's not supported
            if api_provider == "OpenAI":
                st.info("ℹ️ Top K is not supported by OpenAI models and will be ignored")
                top_k = (
                    st.session_state.param_top_k
                )  # Keep the value for display purposes
            else:
                top_k = st.slider(
                    "Top K",
                    min_value=1,
                    max_value=100,
                    value=st.session_state.param_top_k,
                    step=1,
                    key="top_k_slider",
                    help="Limits the number of tokens the model considers for each generation step. Only the K most likely tokens are considered, regardless of their cumulative probability. Lower values (1-10) make responses more predictable, while higher values (40-100) allow more variety. This parameter works alongside temperature and top_p to control output diversity.",
                )
                st.session_state.param_top_k = top_k

        with tab2:
            st.subheader("Sampling Parameters")

            # Presence Penalty
            if "param_presence_penalty" not in st.session_state:
                st.session_state.param_presence_penalty = 0.0
            presence_penalty = st.slider(
                "Presence Penalty",
                min_value=-2.0,
                max_value=2.0,
                value=st.session_state.param_presence_penalty,
                step=0.1,
                key="presence_penalty_slider",
                help="Penalizes new tokens based on whether they appear in the text so far. Positive values (0.1-2.0) discourage the model from repeating tokens that have already appeared, making responses more diverse. Negative values (-0.1 to -2.0) encourage repetition. Useful for reducing repetitive phrases and encouraging more varied vocabulary.",
            )
            st.session_state.param_presence_penalty = presence_penalty

            # Frequency Penalty
            if "param_frequency_penalty" not in st.session_state:
                st.session_state.param_frequency_penalty = 0.0
            frequency_penalty = st.slider(
                "Frequency Penalty",
                min_value=-2.0,
                max_value=2.0,
                value=st.session_state.param_frequency_penalty,
                step=0.1,
                key="frequency_penalty_slider",
                help="Penalizes new tokens based on their frequency in the text so far. Positive values (0.1-2.0) reduce the likelihood of frequently used words, encouraging the model to use less common vocabulary. Negative values (-0.1 to -2.0) encourage the use of common words. Works well with presence penalty to create more varied and natural-sounding text.",
            )
            st.session_state.param_frequency_penalty = frequency_penalty

            # Seed (OpenAI only)
            if api_provider == "OpenAI":
                seed = st.number_input(
                    "Seed",
                    min_value=None,
                    max_value=None,
                    value=None,
                    step=1,
                    help="Random seed for reproducible results. When set, the model will generate the same output for identical inputs, making experiments more reliable. Useful for debugging, testing, or when you need consistent outputs. Leave empty for random behavior.",
                )

        with tab3:
            st.subheader("Token Parameters")

            # Max Tokens
            if "param_max_tokens" not in st.session_state:
                st.session_state.param_max_tokens = 1000
            max_tokens = st.slider(
                "Max Tokens",
                min_value=1,
                max_value=4000,
                value=st.session_state.param_max_tokens,
                step=1,
                key="max_tokens_slider",
                help="Maximum number of tokens to generate in the response. Lower values (100-500) create shorter, more concise responses, while higher values (1000-4000) allow for longer, more detailed explanations. Be mindful of model context limits and cost considerations. Note: The actual response may be shorter if the model reaches a natural stopping point.",
            )
            st.session_state.param_max_tokens = max_tokens

            # Stop Sequences (Anthropic)
            if api_provider == "Anthropic":
                stop_sequences = st.text_area(
                    "Stop Sequences (one per line)",
                    placeholder="Enter stop sequences, one per line...",
                    height=80,
                    help="Sequences that will stop generation when encountered. Enter one sequence per line. For example, entering 'END' will make the model stop generating when it encounters the word 'END'. Useful for controlling response length, preventing the model from continuing beyond certain points, or creating structured outputs. Common examples include '###', 'END', or custom markers.",
                )

        with tab4:
            st.subheader("Advanced Parameters")

            # User
            user = st.text_input(
                "User ID",
                placeholder="Optional user identifier",
                help="A unique identifier representing your end-user. This can be used for tracking, analytics, or to help the model provide more personalized responses. The identifier is stored by the API provider and can be used to identify patterns in usage or to implement user-specific features. Examples: 'user_123', 'session_abc', or any string that helps identify the user.",
            )

            # Metadata (Anthropic)
            if api_provider == "Anthropic":
                metadata = st.text_area(
                    "Metadata (JSON)",
                    placeholder='{"key": "value"}',
                    height=80,
                    help="Optional metadata object that can be attached to the API call. This metadata is stored by Anthropic and can be used for analytics, debugging, or custom processing. Must be valid JSON format. Examples: {'project': 'my_app', 'version': '1.0', 'environment': 'production'} or {'user_id': '123', 'session_id': 'abc'}.",
                )

        with tab5:
            st.subheader("Response Format")

            # Response Format (OpenAI)
            if api_provider == "OpenAI":
                response_format_type = st.selectbox(
                    "Response Format",
                    ["auto", "text", "json_object"],
                    help="Specify the format of the response. 'auto' lets the model choose the most appropriate format. 'text' ensures plain text output. 'json_object' forces the model to return valid JSON, useful for structured data extraction or when you need parseable responses. Note: When using 'json_object', the model will always attempt to return valid JSON, even if it means the response might be less natural.",
                )

                if response_format_type == "json_object":
                    response_format = {"type": "json_object"}
                elif response_format_type == "text":
                    response_format = {"type": "text"}
                else:
                    response_format = None

            # Tools (OpenAI)
            if api_provider == "OpenAI":
                st.subheader("Tools")
                use_tools = st.checkbox(
                    "Enable Tools",
                    help="Enable function calling capabilities. When enabled, the model can call predefined functions (tools) to perform specific tasks like retrieving data, making calculations, or interacting with external systems. This allows the model to access real-time information and perform actions beyond text generation.",
                )

                if use_tools:
                    tool_choice = st.selectbox(
                        "Tool Choice",
                        ["auto", "none", "required"],
                        help="Controls how the model responds to function calls. 'auto' lets the model decide whether to use tools based on the conversation. 'none' prevents the model from using any tools. 'required' forces the model to use at least one tool if available. Useful for controlling when and how the model should interact with external functions.",
                    )

                    # Simple tool example
                    tools = [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get the current weather in a given location",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "location": {
                                            "type": "string",
                                            "description": "The city and state, e.g. San Francisco, CA",
                                        },
                                        "unit": {
                                            "type": "string",
                                            "enum": ["celsius", "fahrenheit"],
                                        },
                                    },
                                    "required": ["location"],
                                },
                            },
                        }
                    ]
                else:
                    tools = None
                    tool_choice = None

    with col2:
        st.header("📤 API Payload")

        # Prepare messages
        messages = []
        if user_message.strip():
            messages.append({"role": "user", "content": user_message})

        # Create payload based on provider
        if api_provider == "OpenAI":
            # For OpenAI, include system message in messages array
            messages_with_system = []
            if system_message.strip():
                messages_with_system.append(
                    {"role": "system", "content": system_message}
                )
            messages_with_system.extend(messages)

            payload = create_openai_payload(
                model=selected_model,
                messages=messages_with_system,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                user=user if user else None,
                seed=seed if "seed" in locals() else None,
                response_format=(
                    response_format if "response_format" in locals() else None
                ),
                tools=tools if "tools" in locals() else None,
                tool_choice=tool_choice if "tool_choice" in locals() else None,
            )
        else:  # Anthropic
            # Parse stop sequences
            stop_sequences_list = None
            if "stop_sequences" in locals() and stop_sequences.strip():
                stop_sequences_list = [
                    seq.strip() for seq in stop_sequences.split("\n") if seq.strip()
                ]

            # Parse metadata
            metadata_obj = None
            if "metadata" in locals() and metadata.strip():
                try:
                    metadata_obj = json.loads(metadata)
                except json.JSONDecodeError:
                    st.error("Invalid JSON in metadata")

            payload = create_anthropic_payload(
                model=selected_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                system=system_message if system_message.strip() else None,
                metadata=metadata_obj,
                stop_sequences=stop_sequences_list,
            )

        # Display payload
        st.json(payload)

        # Make API call if button was clicked
        if st.session_state.get("generate_clicked", False):
            st.session_state.generate_clicked = False

            with st.spinner("Making API call..."):
                if api_provider == "OpenAI":
                    result = call_openai_api(payload)
                else:
                    result = call_anthropic_api(payload)

                # Store result
                st.session_state.api_responses.append(
                    {
                        "timestamp": datetime.now(),
                        "provider": api_provider,
                        "payload": payload,
                        "result": result,
                    }
                )

        # Display latest response
        if st.session_state.api_responses:
            st.header("📥 Latest Response")
            latest_response = st.session_state.api_responses[-1]

            if latest_response["result"]["success"]:
                st.success("✅ API Call Successful")

                # Display response content
                if latest_response["provider"] == "OpenAI":
                    content = latest_response["result"]["choices"][0]["message"][
                        "content"
                    ]
                else:  # Anthropic
                    content = latest_response["result"]["content"][0]["text"]

                st.text_area("Response Content", content, height=200)

                # Display usage information and costs
                if latest_response["result"]["usage"]:
                    usage = latest_response["result"]["usage"]

                    if latest_response["provider"] == "OpenAI":
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        total_tokens = usage.get("total_tokens", 0)

                        # Calculate costs
                        cost_result = calculate_cost(
                            latest_response["provider"],
                            selected_model,
                            prompt_tokens,
                            completion_tokens,
                        )

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Prompt Tokens", prompt_tokens)
                        with col2:
                            st.metric("Completion Tokens", completion_tokens)
                        with col3:
                            st.metric("Total Tokens", total_tokens)
                        with col4:
                            if cost_result["success"]:
                                st.metric(
                                    "Total Cost", f"${cost_result['total_cost']:.4f}"
                                )
                            else:
                                st.metric("Cost", "N/A")

                        # Display detailed cost breakdown
                        if cost_result["success"]:
                            with st.expander("Cost Breakdown"):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric(
                                        "Input Cost",
                                        f"${cost_result['input_cost']:.4f}",
                                    )
                                    st.caption(
                                        f"${cost_result['input_cost_per_1k']:.4f}/1K tokens"
                                    )
                                with col2:
                                    st.metric(
                                        "Output Cost",
                                        f"${cost_result['output_cost']:.4f}",
                                    )
                                    st.caption(
                                        f"${cost_result['output_cost_per_1k']:.4f}/1K tokens"
                                    )
                                with col3:
                                    st.metric(
                                        "Total Cost",
                                        f"${cost_result['total_cost']:.4f}",
                                    )
                                    st.caption("Input + Output")
                    else:  # Anthropic
                        input_tokens = usage.get("input_tokens", 0)
                        output_tokens = usage.get("output_tokens", 0)
                        cache_creation_tokens = usage.get(
                            "cache_creation_input_tokens", 0
                        )
                        cache_read_tokens = usage.get("cache_read_input_tokens", 0)

                        # Calculate costs
                        cost_result = calculate_cost(
                            latest_response["provider"],
                            selected_model,
                            input_tokens,
                            output_tokens,
                        )

                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Input Tokens", input_tokens)
                        with col2:
                            st.metric("Output Tokens", output_tokens)
                        with col3:
                            st.metric("Cache Creation Tokens", cache_creation_tokens)
                        with col4:
                            st.metric("Cache Read Tokens", cache_read_tokens)
                        with col5:
                            if cost_result["success"]:
                                st.metric(
                                    "Total Cost", f"${cost_result['total_cost']:.4f}"
                                )
                            else:
                                st.metric("Cost", "N/A")

                        # Display detailed cost breakdown
                        if cost_result["success"]:
                            with st.expander("Cost Breakdown"):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric(
                                        "Input Cost",
                                        f"${cost_result['input_cost']:.4f}",
                                    )
                                    st.caption(
                                        f"${cost_result['input_cost_per_1k']:.4f}/1K tokens"
                                    )
                                with col2:
                                    st.metric(
                                        "Output Cost",
                                        f"${cost_result['output_cost']:.4f}",
                                    )
                                    st.caption(
                                        f"${cost_result['output_cost_per_1k']:.4f}/1K tokens"
                                    )
                                with col3:
                                    st.metric(
                                        "Total Cost",
                                        f"${cost_result['total_cost']:.4f}",
                                    )
                                    st.caption("Input + Output")

                # Display response details
                with st.expander("Response Details"):
                    st.json(latest_response["result"])
            else:
                st.error("❌ API Call Failed")
                st.error(latest_response["result"]["error"])

    # Response History
    if len(st.session_state.api_responses) > 1:
        st.header("📚 Response History")

        for i, response in enumerate(reversed(st.session_state.api_responses[:-1])):
            with st.expander(
                f"Response {len(st.session_state.api_responses) - i - 1} - {response['timestamp'].strftime('%H:%M:%S')}"
            ):
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("Payload")
                    st.json(response["payload"])

                with col2:
                    st.subheader("Result")
                    if response["result"]["success"]:
                        st.success("✅ Success")
                        if response["provider"] == "OpenAI":
                            content = response["result"]["choices"][0]["message"][
                                "content"
                            ]
                        else:  # Anthropic
                            content = response["result"]["content"][0]["text"]
                        st.text_area("Content", content, height=100, key=f"history_{i}")

                        # Display usage and cost information for history
                        if response["result"]["usage"]:
                            usage = response["result"]["usage"]

                            # Get the model from the payload
                            model = response["payload"]["model"]

                            if response["provider"] == "OpenAI":
                                prompt_tokens = usage.get("prompt_tokens", 0)
                                completion_tokens = usage.get("completion_tokens", 0)
                                total_tokens = usage.get("total_tokens", 0)

                                cost_result = calculate_cost(
                                    response["provider"],
                                    model,
                                    prompt_tokens,
                                    completion_tokens,
                                )

                                # Display token breakdown
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Prompt Tokens", prompt_tokens)
                                with col2:
                                    st.metric("Completion Tokens", completion_tokens)
                                with col3:
                                    st.metric("Total Tokens", total_tokens)
                                with col4:
                                    if cost_result["success"]:
                                        st.metric(
                                            "Total Cost",
                                            f"${cost_result['total_cost']:.4f}",
                                        )
                                    else:
                                        st.metric("Cost", "N/A")

                                # Display detailed cost breakdown
                                if cost_result["success"]:
                                    st.subheader("Cost Breakdown")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric(
                                            "Input Cost",
                                            f"${cost_result['input_cost']:.4f}",
                                        )
                                        st.caption(
                                            f"${cost_result['input_cost_per_1k']:.4f}/1K tokens"
                                        )
                                    with col2:
                                        st.metric(
                                            "Output Cost",
                                            f"${cost_result['output_cost']:.4f}",
                                        )
                                        st.caption(
                                            f"${cost_result['output_cost_per_1k']:.4f}/1K tokens"
                                        )
                                    with col3:
                                        st.metric(
                                            "Total Cost",
                                            f"${cost_result['total_cost']:.4f}",
                                        )
                                        st.caption("Input + Output")
                            else:  # Anthropic
                                input_tokens = usage.get("input_tokens", 0)
                                output_tokens = usage.get("output_tokens", 0)
                                cache_creation_tokens = usage.get(
                                    "cache_creation_input_tokens", 0
                                )
                                cache_read_tokens = usage.get(
                                    "cache_read_input_tokens", 0
                                )

                                cost_result = calculate_cost(
                                    response["provider"],
                                    model,
                                    input_tokens,
                                    output_tokens,
                                )

                                # Display token breakdown
                                col1, col2, col3, col4, col5 = st.columns(5)
                                with col1:
                                    st.metric("Input Tokens", input_tokens)
                                with col2:
                                    st.metric("Output Tokens", output_tokens)
                                with col3:
                                    st.metric(
                                        "Cache Creation Tokens", cache_creation_tokens
                                    )
                                with col4:
                                    st.metric("Cache Read Tokens", cache_read_tokens)
                                with col5:
                                    if cost_result["success"]:
                                        st.metric(
                                            "Total Cost",
                                            f"${cost_result['total_cost']:.4f}",
                                        )
                                    else:
                                        st.metric("Cost", "N/A")

                                # Display detailed cost breakdown
                                if cost_result["success"]:
                                    st.subheader("Cost Breakdown")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric(
                                            "Input Cost",
                                            f"${cost_result['input_cost']:.4f}",
                                        )
                                        st.caption(
                                            f"${cost_result['input_cost_per_1k']:.4f}/1K tokens"
                                        )
                                    with col2:
                                        st.metric(
                                            "Output Cost",
                                            f"${cost_result['output_cost']:.4f}",
                                        )
                                        st.caption(
                                            f"${cost_result['output_cost_per_1k']:.4f}/1K tokens"
                                        )
                                    with col3:
                                        st.metric(
                                            "Total Cost",
                                            f"${cost_result['total_cost']:.4f}",
                                        )
                                        st.caption("Input + Output")
                    else:
                        st.error("❌ Failed")
                        st.error(response["result"]["error"])

    # Cost Summary
    if st.session_state.api_responses:
        st.header("Cost Summary")

        total_cost = 0.0
        total_tokens = 0
        successful_responses = 0

        for response in st.session_state.api_responses:
            if response["result"]["success"] and response["result"]["usage"]:
                usage = response["result"]["usage"]
                model = response["payload"]["model"]

                if response["provider"] == "OpenAI":
                    input_tokens = usage.get("prompt_tokens", 0)
                    output_tokens = usage.get("completion_tokens", 0)
                else:  # Anthropic
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)

                cost_result = calculate_cost(
                    response["provider"], model, input_tokens, output_tokens
                )

                if cost_result["success"]:
                    total_cost += cost_result["total_cost"]
                    total_tokens += input_tokens + output_tokens
                    successful_responses += 1

        if successful_responses > 0:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Responses", len(st.session_state.api_responses))
            with col2:
                st.metric("Successful Calls", successful_responses)
            with col3:
                st.metric("Total Tokens", f"{total_tokens:,}")
            with col4:
                st.metric("Total Cost", f"${total_cost:.4f}")
        else:
            st.info("No successful API calls with cost data available")


if __name__ == "__main__":
    main()
