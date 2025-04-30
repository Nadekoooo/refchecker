from django import forms

class PaperUploadForm(forms.Form):
    paper1 = forms.FileField(label="Paper 1 (file .txt)")
    paper2 = forms.FileField(label="Paper 2 (file .txt)")

    # Optional: add clean method for custom validation if needed
    def clean(self):
        cleaned_data = super().clean()
        f1 = cleaned_data.get('paper1')
        f2 = cleaned_data.get('paper2')
        if not f1 or not f2:
            raise forms.ValidationError("Dua file harus diunggah.")
        return cleaned_data
