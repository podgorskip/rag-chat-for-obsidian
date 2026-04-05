class Config:
  def __init__(self):
    self.MAX_CONTEXT_TOKENS = 2000
    self.MIN_SIMILARITY = 0.5
    self.DELTA_CUTOFF = 0.08
    self.TOP_K = 0.5
    self.REPHRASE_PROMPT = ("You are a helpful assistant. Rephrase the following user query to be more descriptive and "
                            "search-engine friendly. Output ONLY the rephrased query.")
    self.ANSWER_PROMPT = ("You are a helpful assistant answering questions. Use provided passages if available. "
                          "If the passages are not relevant to the question, you may ignore them. If you don't know "
                          "the answer, say you don't know. Be concise and don't use 'The passage...'")