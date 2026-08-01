from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

class ModelRouter:
    """Route queries to appropriate model based on complexity."""

    def __init__(self):
        self.cheap_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        self.expensive_model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
        self.classifier = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    def classify_complexity(self, query: str) -> str:
        """Classify query complexity."""

        prompt = ChatPromptTemplate.from_template(
            """
            Classify this query's complexity as 'simple' or 'complex'.

            Simple: Basic facts, short answers, simple calculations
            Complex: Analysis, reasoning, creative tasks, multi-step problems

            Query: {query}

            Respond with only: simple or complex
        """
        )

        response = self.classifier.invoke(prompt.format(query=query))
        return response.text.strip().lower()

    @traceable(name="routed_query")
    def invoke(self, query: str) -> tuple[str, str, float]:
        """
        Route and invoke query.
        Returns: (response, model_used, estimated_cost)
        """
        complexity = self.classify_complexity(query)

        if complexity == "simple":
            model = self.cheap_model
            model_name = "gemini-2.5-flash"
            cost_per_1k = 0.00015  # Input cost
        else:
            model = self.expensive_model
            model_name = "gemini-3.5-flash"
            cost_per_1k = 0.0025  # Input cost

        response = model.invoke(query)

        # Estimate cost (rough)
        tokens = len(query.split()) * 1.4
        estimated_cost = (tokens / 1000) * cost_per_1k

        return response.text, model_name, estimated_cost
    
    
def demo_model_routing():
    router = ModelRouter()

    queries = [
        "What is 5*5?",  
        "what is epistemic uncertainty? how does it affect machine learning models?", 
        "What is the capital of India?",  
    ]

    print("Model Routing Demo:\n")

    total_cost = 0
    for query in queries:
        result, model, cost = router.invoke(query)
        total_cost += cost
        print(f"Query: {query[:]}...")
        print(f"  Model: {model}")
        print(f"  Est. Cost: ${cost:.6f}")
        print(f"  Response: {result[:]}...")

    print(f"\nTotal Estimated Cost: ${total_cost:.6f}")
    
    

class TokenBudget:
    """Track and limit token usage."""

    def __init__(self, max_tokens_per_request: int = 4000):
        self.max_per_request = max_tokens_per_request
        self.usage = {"total_input": 0, "total_output": 0, "requests": 0}

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (actual would use tiktoken)."""
        return int(len(text.split()) * 1.4)

    def check_budget(self, text: str) -> tuple[bool, int]:
        """Check if request is within budget."""
        tokens = self.estimate_tokens(text)
        return tokens <= self.max_per_request, tokens

    def record_usage(self, input_tokens: int, output_tokens: int):
        """Record token usage."""
        self.usage["total_input"] += input_tokens
        self.usage["total_output"] += output_tokens
        self.usage["requests"] += 1

    def get_stats(self) -> dict:
        return {
            **self.usage,
            "total_tokens": self.usage["total_input"] + self.usage["total_output"],
            "avg_per_request": (
                (self.usage["total_input"] + self.usage["total_output"])
                / max(self.usage["requests"], 1)
            ),
        }
        
        
class BudgetedLLM:
    """LLM with token budgeting."""

    def __init__(self, max_tokens: int = 4000):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
        self.budget = TokenBudget(max_tokens_per_request=max_tokens)

    @traceable(name="budgeted_invoke")
    def invoke(self, query: str) -> str:
        # Check budget
        within_budget, tokens = self.budget.check_budget(query)

        if not within_budget:
            raise ValueError(
                f"Query exceeds token budget: {tokens} > {self.budget.max_per_request}"
            )

        response = self.llm.invoke(query)
        result = response.text

        output_tokens = self.budget.estimate_tokens(result)
        self.budget.record_usage(tokens, output_tokens)

        return result

    def get_stats(self) -> dict:
        return self.budget.get_stats()


def demo_token_budgeting():

    llm = BudgetedLLM(max_tokens=100)

    queries = [
        "What is AI?",  # Within budget
        "Explain " + "very " * 100 + "complex topic",  # Over budget
    ]

    print("\nToken Budgeting Demo:\n")

    for query in queries:
        try:
            result = llm.invoke(query)
            print(f"{query[:40]}... -> {result[:30]}...")
        except ValueError as e:
            print(f"{query[:40]}... -> {e}")

    print(f"\nUsage: {llm.get_stats()}")


if __name__ == "__main__":
    #demo_model_routing()
    demo_token_budgeting()