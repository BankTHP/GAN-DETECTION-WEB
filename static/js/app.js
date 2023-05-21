const buttons = document.querySelectorAll("button");
for (let i = 0; i < buttons.length; i++) {
    const buttonText = buttons[i].textContent;
	buttons[i].addEventListener("click", function(){
		window.alert(`${buttonText}`);
	});
}