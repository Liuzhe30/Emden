function submit1() {
// alert("Submitted successfully!");
}

function demo_download() {
    alert('download!');
    window.open("/static/file/eg_input.txt", 'blank');
}

function formReset() {
    document.getElementsByName('form1').reset()
}

function formReset2() {
    document.getElementsByName('form2').reset()
}

function example() {
    form1.seq.value = "MSDVAIVKEGWLHKRGEYIKTWRPRYFLLKNDGTFIGYKERPQDVDQREAPLNNFSVAQCQLMKTERPRPNTF";
    form1.muinfo.value = "E17K";
    form1.drugname.value = "Vemurafenib";
    form1.form1_submit.focus();
}

function example2() {
    form2.pdb_id.value = "1CSE";
    form2.mutation_chain.value = "I";
    form2.mut_2.value = "L45G";
    form2.partner_chain.value = "E";
    form2.form2_submit.focus();
}

function submit2() {
// alert("Submitted successfully!");
}

function submit3(){
    alert('It is not online and is under urgent development, so stay tuned !');
}