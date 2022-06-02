import "./ExpensesList.css";
import ExpenseItem from "./ExpenseItem";

const ExpensesList = (props) => {

  if (props.filteredExpenses.length === 0) {
      return <h2 className='expenses-list__fallback'>Found No Expenses.</h2>
  }
    return (
      <ul className="expenses-list">
        {props.filteredExpenses.map((expense) => (
          <ExpenseItem
            key={expense.id}
            id={expense.id}
            title={expense.title}
            amount={expense.amount}
            date={expense.date}
            onDelete={props.onDelete}
          />
        ))}
      </ul>
    );
};

export default ExpensesList;
