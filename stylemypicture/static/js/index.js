var loadFile = function (event) {
	var image = document.getElementById("loaded_image");
	image.src = URL.createObjectURL(event.target.files[0]);
};
/*
var submitForms = function () {
	document.getElementById(form1).submit();
	document.getElementById(form2).submit();
};
*/
var submit_form = function () {

	document.getElementById(form1).submit();
}
var myCarousel = document.getElementById("carouselExampleControls");
var carousel = bootstrap.Carousel.getInstance(myCarousel);
var slide_number = 0;
myCarousel.addEventListener("slide.bs.carousel", function (event) {
	// do something...
	//var a= carousel.from(event.relatedTarget).index;
	/*
				e.direction     // The direction in which the carousel is sliding (either "left" or "right").
			  e.relatedTarget // The DOM element that is being slid into place as the active item.
			e.from          // The index of the current item.     
			  e.to            // The index of the next item.
	*/
	if (event.direction == "left") {
		slide_number = slide_number + 1;
		if (slide_number == 21) {
			slide_number = 0;
		}
	}
	if (event.direction == "right") {
		slide_number = slide_number - 1;
		if (slide_number == -1) {
			slide_number = 20;
		}
	}
	var a = event.from;
	document.getElementById("slideNumber").value = slide_number;
	console.log(slide_number);
});

/*
var abcd =  function () {
	
	var myCarousel = document.querySelector('#carouselExampleControls')
	myCarousel.next();
	
	//console.log(mySpecialCarouselInit)
   };

 */
