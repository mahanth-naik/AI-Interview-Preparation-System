export const initialForm = Object.freeze({
  role: "",
  interview_type: "technical",
  difficulty: "medium",
  number_of_questions: 5,
});

export function createAccountRequestGuard() {
  let generation = 0;
  let accountId = null;

  function capture() {
    return { generation, accountId };
  }

  function isCurrent(context) {
    return (
      context?.accountId != null &&
      context.generation === generation &&
      context.accountId === accountId
    );
  }

  return {
    activate(nextAccountId) {
      if (nextAccountId == null) {
        throw new TypeError("An account ID is required.");
      }
      generation += 1;
      accountId = nextAccountId;
      return capture();
    },
    invalidate() {
      generation += 1;
      accountId = null;
      return capture();
    },
    capture,
    isCurrent,
    isGenerationCurrent(context) {
      return context?.generation === generation;
    },
    async run(context, request, apply) {
      if (!isCurrent(context)) return false;
      const result = await request();
      if (!isCurrent(context)) return false;
      apply(result);
      return true;
    },
  };
}

export function clearInterviewState({
  setSession,
  setQuestion,
  setAnswer,
  setEvaluation,
  setReport,
}) {
  setSession(null);
  setQuestion(null);
  setAnswer("");
  setEvaluation(null);
  setReport(null);
}

export function clearAccountState(setters) {
  clearInterviewState(setters);
  setters.setSessions([]);
  setters.setDocumentId("");
  setters.setDocuments([]);
  setters.setForm(initialForm);
  setters.setDocumentType("resume");
}
